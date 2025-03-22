import json
import streamlit as st
import os
import zipfile
import io
import uuid
import itertools
import torch
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision
from google.cloud import translate_v2 as translate
from pypinyin import lazy_pinyin, Style
import jieba
from transformers import AutoTokenizer, AutoModel

# ------------------ LOAD GOOGLE CLOUD CREDENTIALS FROM SECRETS ------------------
if "google_cloud" in st.secrets:
    creds_dict = dict(st.secrets["google_cloud"])  # Already a dict
    creds_path = "temp_google_credentials.json"
    with open(creds_path, "w") as f:
        json.dump(creds_dict, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# ------------------ HELPER FUNCTIONS ------------------
def is_all_korean(text):
    """
    Returns True if all alphanumeric tokens in text are purely Korean.
    """
    tokens = re.findall(r'\w+', text)
    if not tokens:
        return False
    for token in tokens:
        if not re.fullmatch(r'[\uac00-\ud7a3]+', token):
            return False
    return True

def filter_korean(text):
    """
    Removes tokens that contain any Korean characters.
    """
    tokens = text.split()
    filtered_tokens = [tok for tok in tokens if not re.search(r'[\uac00-\ud7a3]', tok)]
    return " ".join(filtered_tokens)



def detect_text_boxes_document(image_path):
    response = vision_client.document_text_detection(...)
    if not response.full_text_annotation:
        return []

    annotations = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            # Combine all paragraphs in this block
            block_text = []
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_str = "".join([s.text for s in word.symbols])
                    block_text.append(word_str)

            block_text_str = " ".join(block_text).strip()

            vs = block.bounding_box.vertices
            xs = [v.x for v in vs]
            ys = [v.y for v in vs]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            annotations.append({
                "bbox": (min_x, min_y, max_x, max_y),
                "text": block_text_str
            })
    return annotations



def split_into_sentences(text):
    """
    Naive approach to split text into sentences based on '.', '?', and '!' delimiters.
    Preserves the delimiter at the end of each sentence.
    """
    parts = re.split(r'([.?!])', text)
    sentences = []
    for i in range(0, len(parts), 2):
        raw_sent = parts[i].strip()
        if i + 1 < len(parts):
            raw_sent += parts[i+1]  # attach punctuation
        raw_sent = raw_sent.strip()
        if raw_sent:
            sentences.append(raw_sent)
    return sentences

# ------------------ GOOGLE CLIENTS ------------------
translation_client = translate.Client()
vision_client = vision.ImageAnnotatorClient()

def google_translate(text_en):
    """
    Translates English text to Chinese (zh-CN) using Google Translate.
    Returns tuple: (chinese_text, alignment_placeholder).
    """
    text_en = text_en.strip()
    if not text_en:
        return "", ""
    try:
        result = translation_client.translate(
            text_en,
            target_language='zh-CN',
            source_language='en'
        )
        cn_text = result['translatedText']
        return cn_text, ""
    except Exception as e:
        st.write(f"Google translation error: {e}")
        return text_en, ""

def translate_text(english_text):
    """
    Simple wrapper to get only the Chinese text.
    """
    cn_text, _ = google_translate(english_text)
    return cn_text

# ------------------ FONTS & WRAPPING ------------------
FONT_PATH = "NotoSans-Regular.ttf"  # Provide a valid path to your font
MARGIN = 5
MERGE_THRESHOLD = 20

def get_text_size(text, font):
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        return font.getmask(text).size

def wrap_tokens(tokens, font, max_width):
    """
    Wraps (word, color) tokens into lines within max_width.
    Returns (lines, total_height).
    """
    space_width, _ = get_text_size(" ", font)
    lines = []
    current_line = []
    current_width = 0

    for token_text, token_color in tokens:
        token_width, _ = get_text_size(token_text, font)
        if current_line:
            if current_width + space_width + token_width > max_width:
                lines.append(current_line)
                current_line = [(token_text, token_color)]
                current_width = token_width
            else:
                current_line.append((token_text, token_color))
                current_width += space_width + token_width
        else:
            current_line.append((token_text, token_color))
            current_width = token_width

    if current_line:
        lines.append(current_line)

    line_height = get_text_size("Ay", font)[1]
    total_height = line_height * len(lines)
    return lines, total_height

def draw_wrapped_lines(draw, lines, font, start_x, start_y, max_width):
    """
    Draw lines of (word, color) tokens, centered horizontally in max_width.
    """
    space_width, _ = get_text_size(" ", font)
    line_height = get_text_size("Ay", font)[1]
    for line in lines:
        line_width = sum(get_text_size(t[0], font)[0] for t in line) + space_width*(len(line)-1)
        line_x = start_x + (max_width - line_width) / 2
        x = line_x
        for token_text, token_color in line:
            draw.text((x, start_y), token_text, font=font, fill=token_color)
            token_width, _ = get_text_size(token_text, font)
            x += token_width + space_width
        start_y += line_height
    return start_y



def try_expand_box(orig_box, other_boxes, img_width, img_height,
                   expand_w_factor=0.2, expand_h_factor=0.35,
                   margin=5, extra_padding=10):
    """
    Attempt to expand orig_box by (expand_w_factor * 100)% in width
    and (expand_h_factor * 100)% in height, plus margin and extra_padding.
    
    For a single bounding box scenario, other_boxes can be [].
    """
    (min_x, min_y, max_x, max_y) = orig_box

    # Basic expansions for margin + padding
    min_x = max(0, min_x - margin - extra_padding)
    min_y = max(0, min_y - margin - extra_padding)
    max_x = min(img_width, max_x + margin + extra_padding)
    max_y = min(img_height, max_y + margin + extra_padding)

    original_expanded = (min_x, min_y, max_x, max_y)

    # Expand around center
    width = max_x - min_x
    height = max_y - min_y
    expand_w = width * expand_w_factor
    expand_h = height * expand_h_factor

    new_min_x = max(0, min_x - expand_w / 2)
    new_max_x = min(img_width, max_x + expand_w / 2)
    new_min_y = max(0, min_y - expand_h / 2)
    new_max_y = min(img_height, max_y + expand_h / 2)

    # Since we’re using a single bounding box, other_boxes is empty,
    # so we can just return the expanded box directly.
    return (new_min_x, new_min_y, new_max_x, new_max_y)


# ------------------ CBZ HANDLING ------------------
def extract_cbz(cbz_path, output_folder):
    with zipfile.ZipFile(cbz_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    images = []
    for f in sorted(os.listdir(output_folder)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(output_folder, f))
    return images

def repack_to_cbz(folder_path, output_cbz_path):
    with zipfile.ZipFile(output_cbz_path, "w") as zf:
        for root, _, files in os.walk(folder_path):
            for file in sorted(files):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zf.write(full_path, arcname=rel_path)

# ------------------ SINGLE BOX DETECTION USING text_annotations[0] ------------------
def bbox_for_annotation(ann):
    """
    Takes a Vision API annotation object (not a dict) and returns (min_x, min_y, max_x, max_y).
    """
    vs = ann.bounding_poly.vertices
    xs = [v.x for v in vs]
    ys = [v.y for v in vs]
    return min(xs), min(ys), max(xs), max(ys)

def detect_text_boxes(image_path):
    """
    Document Text Detection approach returning multiple bounding boxes
    (one per paragraph).
    """
    with open(image_path, "rb") as img_file:
        content = img_file.read()
    image = vision.Image(content=content)

    response = vision_client.document_text_detection(image=image)
    if not response.full_text_annotation:
        return []

    annotations = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                # build paragraph text
                para_text = ""
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    para_text += word_text + " "
                para_text = para_text.strip()

                vs = paragraph.bounding_box.vertices
                xs = [v.x for v in vs]
                ys = [v.y for v in vs]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                annotations.append({"bbox": (x1, y1, x2, y2), "text": para_text})

    return annotations


# ------------------ AWESOME-ALIGN MODEL ------------------
@st.cache_resource
def load_awesome_align_model():
    model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
    tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")
    return model, tokenizer

def awesome_align(english_text, chinese_text):
    align_layer = 8
    th = 1e-3

    sent_src = english_text.strip().split()
    sent_tgt = list(jieba.cut(chinese_text))

    model, tokenizer = load_awesome_align_model()
    token_src = [tokenizer.tokenize(word) for word in sent_src]
    token_tgt = [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
    wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

    ids_src = tokenizer.prepare_for_model(
        list(itertools.chain(*wid_src)),
        return_tensors='pt',
        model_max_length=tokenizer.model_max_length,
        truncation=True
    )['input_ids']
    ids_tgt = tokenizer.prepare_for_model(
        list(itertools.chain(*wid_tgt)),
        return_tensors='pt',
        model_max_length=tokenizer.model_max_length,
        truncation=True
    )['input_ids']

    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i] * len(word_list)
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i] * len(word_list)

    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)
        softmax_inter = (softmax_srctgt > th) * (softmax_tgtsrc > th)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_dict = {}
    for i_idx, j_idx in align_subwords:
        prob = (softmax_srctgt[i_idx, j_idx].item() + softmax_tgtsrc[i_idx, j_idx].item()) / 2
        key = (sub2word_map_src[i_idx], sub2word_map_tgt[j_idx])
        align_dict.setdefault(key, []).append(prob)

    aggregated_alignments = {
        (k[0], k[1], sum(v)/len(v)) for k, v in align_dict.items()
    }
    return str(aggregated_alignments)

# ------------------ MULTI-SENTENCE TRANSLATION & SEGMENTS ------------------
def translate_to_segments(english_text):
    """
    1) Split 'english_text' into multiple sentences.
    2) For each sentence:
       - If it's all Korean, skip.
       - Filter out tokens with Korean, translate leftover, run Awesome-Align.
       - Color-code aligned tokens (English, Chinese, pinyin).
    3) Return (seg_eng, seg_mand, seg_pin), alignment placeholder, combined Chinese text.
    """
    sentence_list = split_into_sentences(english_text)

    all_seg_eng = []
    all_seg_mand = []
    all_seg_pin = []

    normal_palette = ["blue","green","orange","purple","brown","cyan","magenta","olive","teal","navy"]
    color_idx = 0

    for sent in sentence_list:
        if is_all_korean(sent):
            continue

        filtered_text = filter_korean(sent)
        if not filtered_text.strip():
            continue

        cn_text = translate_text(filtered_text)
        mapping_str = awesome_align(filtered_text, cn_text)
        st.write("Awesome-Align mapping (per sentence):", mapping_str)

        try:
            mapping_set = eval(mapping_str)
            mapping_pairs = [tup for tup in mapping_set if len(tup)==3]
        except Exception as e:
            st.write("Error parsing mapping_str:", e)
            mapping_pairs = []

        mapping_dict = {}
        reverse_mapping = {}
        for i,j,_ in mapping_pairs:
            mapping_dict.setdefault(i, set()).add(j)
            reverse_mapping.setdefault(j, set()).add(i)

        src_tokens = filtered_text.split()
        tgt_tokens = list(jieba.cut(cn_text))

        # color the English tokens
        color_mapping = {}
        for i, w in enumerate(src_tokens):
            if i in mapping_dict and color_idx<len(normal_palette):
                color_mapping[i] = normal_palette[color_idx]
                color_idx = (color_idx+1)%len(normal_palette)
            else:
                color_mapping[i] = "black"

        # color the Chinese tokens
        tgt_color = {}
        for j,w in enumerate(tgt_tokens):
            if j in reverse_mapping:
                src_index = list(reverse_mapping[j])[0]
                tgt_color[j] = color_mapping.get(src_index,"black")
            else:
                tgt_color[j] = "black"

        seg_eng = [(w, color_mapping.get(i,"black")) for i,w in enumerate(src_tokens)]
        seg_mand= [(w, tgt_color.get(j,"black"))   for j,w in enumerate(tgt_tokens)]
        seg_pin = [(" ".join(lazy_pinyin(w,style=Style.TONE)), tgt_color.get(j,"black"))
                   for j,w in enumerate(tgt_tokens)]

        if all_seg_eng:
            all_seg_eng.append((" ","black"))
            all_seg_mand.append((" ","black"))
            all_seg_pin.append((" ","black"))

        all_seg_eng.extend(seg_eng)
        all_seg_mand.extend(seg_mand)
        all_seg_pin.extend(seg_pin)

    if not all_seg_eng:
        return None,"", english_text

    combined_chinese = " ".join([tok[0] for tok in all_seg_mand])
    mapping_str = "<multi-sentence alignment>"
    return (all_seg_eng, all_seg_mand, all_seg_pin), mapping_str, combined_chinese

# ------------------ OVERLAY: Single annotation box approach ------------------
def overlay_merged_pinyin(image_path, single_item, font_path=FONT_PATH, margin=MARGIN):
    """
    We have only one annotation item { "bbox":(x1,y1,x2,y2), "text": ... } per image.
    We'll expand it, overlay pinyin, and return final image + text triplets for debug.
    """
    EXTRA_PADDING = 10
    SEPARATOR_PADDING = 10
    initial_font_size = 22
    min_font_size = 14

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    text_triplets = []

    # Expand if desired
    single_item["final_bbox"] = None
    orig_box = single_item["bbox"]
    W,H = img.size

    # We won't merge with others, so pass empty list
    expanded_box = try_expand_box(orig_box, [], W, H,
                                  expand_w_factor=0.2,
                                  expand_h_factor=0.35,
                                  margin=margin,
                                  extra_padding=EXTRA_PADDING)
    single_item["final_bbox"] = expanded_box

    (min_x, min_y, max_x, max_y) = expanded_box
    original_text = single_item["text"].strip()

    seg_result, mapping_str, translated_text = translate_to_segments(original_text)
    if seg_result is None:
        # all-Korean or empty
        return img.convert("RGB"), []

    seg_eng, seg_mand, seg_pin = seg_result
    text_triplets.append((original_text, (seg_eng, seg_mand, seg_pin), mapping_str, translated_text))

    # Draw a white rectangle + red outline
    draw.rectangle([(min_x, min_y), (max_x, max_y)], fill=(255,255,255,255))
    draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=(255,0,0,255), width=2)

    box_width  = max_x - min_x
    box_height = max_y - min_y

    # Font sizing
    font_size = initial_font_size
    while True:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        pinyin_lines, pinyin_height = wrap_tokens(seg_pin, font, box_width)
        english_lines, english_height= wrap_tokens(seg_eng, font, box_width)
        total_text_height = pinyin_height + english_height + SEPARATOR_PADDING

        if total_text_height>box_height and font_size>min_font_size:
            font_size-=2
        else:
            break

    start_y_text = min_y+(box_height-total_text_height)/2
    start_y_text = draw_wrapped_lines(draw, pinyin_lines, font, min_x, start_y_text, box_width)

    start_y_text += SEPARATOR_PADDING
    draw.line([(min_x, start_y_text),(min_x+box_width,start_y_text)], fill="black", width=1)
    start_y_text+=1

    draw_wrapped_lines(draw, english_lines, font, min_x, start_y_text, box_width)

    return img.convert("RGB"), text_triplets

# ------------------ DEBUG PRINT ------------------
def debug_print_ocr_details(image_path):
    """
    Debug: Show before & after images + OCR text, pinyin, alignment.
    """
    from PIL import Image, ImageDraw
    import streamlit as st

    # Original image
    orig_img = Image.open(image_path).convert("RGB")
    before_img = orig_img.copy()
    draw_before = ImageDraw.Draw(before_img)

    # Single annotation (1 bounding box for entire text)
    ann_list = detect_text_boxes(image_path)
    if not ann_list:
        st.write("No OCR text found.")
        return

    ann = ann_list[0]
    box = ann["bbox"]

    # Draw bounding box on the "before" image
    draw_before.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)

    # "After" image with overlay
    after_img, text_triplets = overlay_merged_pinyin(image_path, ann,
                                                     font_path=FONT_PATH, margin=MARGIN)

    st.write("**Before Image (Single bounding box)**")
    st.image(before_img)
    st.write("**After Image (Overlay)**")
    st.image(after_img)

    for idx, triplet in enumerate(text_triplets, start=1):
        original_text, (seg_eng, seg_mand, seg_pin), mapping_str, translated_text = triplet
        st.write(f"--- Debug Info for Annotation {idx} ---")
        st.write("**Extracted OCR Text:**", original_text)
        pin_txt = " ".join([t for t,c in seg_pin])
        st.write("**Pinyin:**", pin_txt)
        st.write("**Word Mapping:**", mapping_str)


# ------------------ STREAMLIT APP ------------------
def main():
    st.title("Batch CBZ Translator (Single BBox per Image)")

    download_placeholder = st.empty()

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    session_id = st.session_state["session_id"]
    base_temp_folder = f"temp_processing_{session_id}"
    output_folder = f"processed_cbz_output_{session_id}"

    if not os.path.exists(base_temp_folder):
        os.makedirs(base_temp_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    st.write("Upload multiple CBZ files.")
    uploaded_files = st.file_uploader("Upload CBZ Files", type=["cbz"], accept_multiple_files=True)

    audio_placeholder = st.empty()
    status_placeholder = st.empty()

    if uploaded_files:
        try:
            with open("1-hour-and-20-minutes-of-silence.mp3","rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_placeholder.audio(audio_bytes, format="audio/mp3")
            status_placeholder.info("Audio is playing...")
        except:
            status_placeholder.warning("Audio file not found. Skipping audio.")

        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        processed_cbz_paths = []

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            cbz_filename = uploaded_file.name
            temp_cbz_path = os.path.join(base_temp_folder, cbz_filename)
            with open(temp_cbz_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            temp_extract_folder = os.path.join(base_temp_folder, f"extract_{uuid.uuid4().hex}")
            os.makedirs(temp_extract_folder, exist_ok=True)
            images = extract_cbz(temp_cbz_path, temp_extract_folder)

            # Process each image
            for img_path in images:
                # Debug if desired
                debug_print_ocr_details(img_path)

                # Final processing with single annotation
                ann_list = detect_text_boxes(img_path)
                if ann_list:
                    final_img, _ = overlay_merged_pinyin(
                        img_path, ann_list[0],
                        font_path=FONT_PATH,
                        margin=MARGIN
                    )
                    final_img.save(img_path)

            # Repack to CBZ
            output_cbz_path = os.path.join(output_folder, cbz_filename)
            repack_to_cbz(temp_extract_folder, output_cbz_path)
            processed_cbz_paths.append(output_cbz_path)

            shutil.rmtree(temp_extract_folder)
            progress_bar.progress(idx/total_files)

        st.success("Processing complete!")

        final_zip = io.BytesIO()
        with zipfile.ZipFile(final_zip,"w") as zf:
            for file in sorted(os.listdir(output_folder)):
                if file.lower().endswith(".cbz"):
                    full_path = os.path.join(output_folder, file)
                    zf.write(full_path, arcname=file)
        final_zip.seek(0)

        st.markdown(
            """
            <script>
            function notifyUser() {
                if (!("Notification" in window)) {
                    console.log("This browser does not support desktop notifications.");
                } else if (Notification.permission === "granted") {
                    new Notification("Processing complete!", { body: "Your final output is ready for download!" });
                } else if (Notification.permission !== "denied") {
                    Notification.requestPermission().then(function(permission) {
                        if (permission === "granted") {
                            new Notification("Processing complete!", { body: "Your final output is ready for download!" });
                        }
                    });
                }
            }
            notifyUser();
            </script>
            """,
            unsafe_allow_html=True
        )

        download_placeholder.download_button(
            label="Download Processed CBZ Files (ZIP)",
            data=final_zip,
            file_name="processed_cbz_files.zip",
            mime="application/zip"
        )

        if st.button("Stop Audio"):
            audio_placeholder.empty()
            status_placeholder.info("Audio stopped.")

    if st.button("Clear My Session Files"):
        shutil.rmtree(base_temp_folder, ignore_errors=True)
        shutil.rmtree(output_folder, ignore_errors=True)
        st.session_state.pop("session_id", None)
        st.experimental_rerun()

if __name__=="__main__":
    main()
