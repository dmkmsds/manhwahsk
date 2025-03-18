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

def tokenize_with_punctuation(text):
    # Insert spaces around any non-alphanumeric/underscore character
    # so that punctuation becomes its own token.
    text = re.sub(r"([^\w\s])", r" \1 ", text)
    # Collapse multiple spaces down to one
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()

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

# ------------------ VISION TEXT DETECTION ------------------
def detect_text_boxes(image_path):
    with open(image_path, "rb") as img_file:
        content = img_file.read()
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    if not response.text_annotations:
        return []
    # text_annotations[0] is the entire recognized text, skip that
    return response.text_annotations[1:]

# ------------------ AWESOME-ALIGN MODEL ------------------
@st.cache_resource
def load_awesome_align_model():
    model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
    tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")
    return model, tokenizer

def awesome_align(english_text, chinese_text):
    """
    Returns a string representation of the aggregated alignments,
    plus a Python set/list of (src_index, tgt_index, probability).
    """
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

    # Average probability for each (src_word_idx, tgt_word_idx)
    aggregated_alignments = []
    for k, v in align_dict.items():
        avg_prob = sum(v)/len(v)
        aggregated_alignments.append((k[0], k[1], avg_prob))

    # Convert to a set for the string representation
    alignment_str_set = {(a[0], a[1], a[2]) for a in aggregated_alignments}
    return str(alignment_str_set), aggregated_alignments

# ------------------ MULTI-SENTENCE TRANSLATION & SEGMENTS ------------------
def translate_to_segments(english_text):
    """
    1) Split 'english_text' into multiple sentences.
    2) For each sentence:
       - If it's all Korean, skip.
       - Filter out tokens with Korean, translate leftover, run Awesome-Align.
       - Color-code aligned tokens (English, Chinese, pinyin).
    3) Return:
       (all_seg_eng, all_seg_mand, all_seg_pin), combined_align_str, combined_cn_text, alignment_info (list of all alignments).
    """
    sentence_list = split_into_sentences(english_text)

    all_seg_eng = []
    all_seg_mand = []
    all_seg_pin = []

    normal_palette = [
        "blue", "green", "orange", "purple", "brown",
        "cyan", "magenta", "olive", "teal", "navy"
    ]
    color_idx = 0

    # We'll accumulate alignment info across sentences
    all_alignment_info = []

    for sent in sentence_list:
        if is_all_korean(sent):
            continue

        filtered_text = filter_korean(sent)
        if not filtered_text.strip():
            continue

        cn_text = translate_text(filtered_text)
        mapping_str, align_list = awesome_align(filtered_text, cn_text)

        # Collect alignment info
        all_alignment_info.extend(align_list)

        # Tokenize English and Chinese
        sent_src = tokenize_with_punctuation(filtered_text)
        sent_tgt = list(jieba.cut(cn_text))

        # Prepare color mappings
        mapping_dict = {}
        reverse_mapping = {}
        for (src_idx, tgt_idx, _) in align_list:
            mapping_dict.setdefault(src_idx, set()).add(tgt_idx)
            reverse_mapping.setdefault(tgt_idx, set()).add(src_idx)

        color_mapping = {}
        for i, _ in enumerate(sent_src):
            if i in mapping_dict and color_idx < len(normal_palette):
                color_mapping[i] = normal_palette[color_idx]
                color_idx = (color_idx + 1) % len(normal_palette)
            else:
                color_mapping[i] = "black"

        target_color_mapping = {}
        for j, _ in enumerate(sent_tgt):
            if j in reverse_mapping:
                # pick any aligned source index
                source_index = list(reverse_mapping[j])[0]
                target_color_mapping[j] = color_mapping.get(source_index, "black")
            else:
                target_color_mapping[j] = "black"

        seg_eng = [(word, color_mapping.get(i, "black")) for i, word in enumerate(sent_src)]
        seg_mand = [(word, target_color_mapping.get(j, "black")) for j, word in enumerate(sent_tgt)]
        seg_pin = [
            (" ".join(lazy_pinyin(word, style=Style.TONE)), target_color_mapping.get(j, "black"))
            for j, word in enumerate(sent_tgt)
        ]

        # Add space between sentences
        if all_seg_eng:
            all_seg_eng.append((" ", "black"))
            all_seg_mand.append((" ", "black"))
            all_seg_pin.append((" ", "black"))

        all_seg_eng.extend(seg_eng)
        all_seg_mand.extend(seg_mand)
        all_seg_pin.extend(seg_pin)

    if not all_seg_eng:
        # Means everything was Korean or empty
        return None, "", english_text, []

    combined_chinese = " ".join([tok[0] for tok in all_seg_mand])
    mapping_str = "<multi-sentence alignment>"
    return (all_seg_eng, all_seg_mand, all_seg_pin), mapping_str, combined_chinese, all_alignment_info

# ------------------ MERGING LOGIC ------------------
def bbox_for_annotation(ann):
    vs = ann.bounding_poly.vertices
    xs = [v.x for v in vs]
    ys = [v.y for v in vs]
    return min(xs), min(ys), max(xs), max(ys)

def overlap_or_close(boxA, boxB, threshold=MERGE_THRESHOLD):
    Aminx, Aminy, Amaxx, Amaxy = boxA
    Bminx, Bminy, Bmaxx, Bmaxy = boxB
    if Amaxx < Bminx - threshold or Bmaxx < Aminx - threshold:
        return False
    if Amaxy < Bminy - threshold or Bmaxy < Aminy - threshold:
        return False
    return True

def merge_boxes_and_text(boxA, boxB, textA, textB):
    Aminx, Aminy, Amaxx, Amaxy = boxA
    Bminx, Bminy, Bmaxx, Bmaxy = boxB
    merged_box = (
        min(Aminx, Bminx), min(Aminy, Bminy),
        max(Amaxx, Bmaxx), max(Amaxy, Bmaxy)
    )
    merged_text = textA + " " + textB
    return merged_box, merged_text

def group_annotations(annotations):
    """
    Merges overlapping text boxes into single items with combined text.
    """
    items = []
    for ann in annotations:
        items.append({"bbox": bbox_for_annotation(ann), "text": ann.description})
    merged = True
    while merged:
        merged = False
        new_items = []
        while items:
            current = items.pop()
            for idx, existing in enumerate(new_items):
                if overlap_or_close(current["bbox"], existing["bbox"], threshold=MERGE_THRESHOLD):
                    mb, mt = merge_boxes_and_text(
                        current["bbox"], existing["bbox"],
                        current["text"], existing["text"]
                    )
                    new_items[idx]["bbox"] = mb
                    new_items[idx]["text"] = mt
                    merged = True
                    break
            else:
                new_items.append(current)
        items = new_items
    return items

# ------------------ BOX OVERLAP (ZERO THRESHOLD) ------------------
def boxes_overlap(boxA, boxB):
    Aminx, Aminy, Amaxx, Amaxy = boxA
    Bminx, Bminy, Bmaxx, Bmaxy = boxB
    if Amaxx < Bminx or Bmaxx < Aminx:
        return False
    if Amaxy < Bminy or Bmaxy < Aminy:
        return False
    return True

# ------------------ EXPAND BOUNDING BOXES ------------------
def try_expand_box(orig_box, other_boxes, img_width, img_height,
                   expand_w_factor=0.2, expand_h_factor=0.35,
                   margin=5, extra_padding=10):
    """
    Attempt to expand `orig_box` by certain width/height factors.
    Revert if it overlaps another box after expansion.
    """
    (min_x, min_y, max_x, max_y) = orig_box

    # Original expansions (margin + padding)
    min_x = max(0, min_x - margin - extra_padding)
    min_y = max(0, min_y - margin - extra_padding)
    max_x = min(img_width, max_x + margin + extra_padding)
    max_y = min(img_height, max_y + margin + extra_padding)

    orig_expanded = (min_x, min_y, max_x, max_y)

    # Now compute the factor expansions around the center
    width = max_x - min_x
    height = max_y - min_y
    expand_w = width * expand_w_factor
    expand_h = height * expand_h_factor

    new_min_x = max(0, min_x - expand_w / 2)
    new_max_x = min(img_width, max_x + expand_w / 2)
    new_min_y = max(0, min_y - expand_h / 2)
    new_max_y = min(img_height, max_y + expand_h / 2)

    expanded_box = (new_min_x, new_min_y, new_max_x, new_max_y)

    # Check overlap with other boxes
    for other in other_boxes:
        if other is None:
            continue
        if other["final_bbox"] is None:
            continue
        if other["final_bbox"] == orig_box:
            continue
        if boxes_overlap(expanded_box, other["final_bbox"]):
            return orig_expanded

    return expanded_box

def expand_boxes(items, img_width, img_height):
    """
    Expand each bounding box if possible. Store in item["final_bbox"].
    """
    for item in items:
        item["final_bbox"] = None

    for i, item in enumerate(items):
        orig_box = item["bbox"]
        expanded_box = try_expand_box(
            orig_box,
            other_boxes=[it for it in items if it is not item],
            img_width=img_width,
            img_height=img_height,
            expand_w_factor=0.2,
            expand_h_factor=0.35,
            margin=MARGIN,
            extra_padding=10
        )
        item["final_bbox"] = expanded_box
    return items

# ------------------ DRAW PREVIEW BOXES (ORIGINAL IMAGE + RED BOX) ------------------
def draw_preview_boxes(image_path, items):
    """
    Returns a new image (RGBA) with only red outline boxes
    around item["final_bbox"] (no fill).
    """
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    for item in items:
        (min_x, min_y, max_x, max_y) = item["final_bbox"]
        draw.rectangle([(min_x, min_y), (max_x, max_y)],
                       outline=(255, 0, 0, 255), width=3)
    return img

# ------------------ OVERLAY TRANSLATION/PINYIN ------------------
def overlay_merged_pinyin(image_path, items, font_path=FONT_PATH, margin=MARGIN):
    """
    Draws white rectangles on final_bbox, then overlays pinyin+English.
    Returns final PIL image, plus a list of dictionaries with detailed info.
    """
    EXTRA_PADDING = 10
    SEPARATOR_PADDING = 10
    initial_font_size = 22
    min_font_size = 14

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # We'll accumulate info about each text region for display
    text_triplets = []

    for item in items:
        (min_x, min_y, max_x, max_y) = item["final_bbox"]
        original_text = item["text"].strip()

        seg_result, mapping_str, translated_text, alignment_info = translate_to_segments(original_text)
        if seg_result is None:
            # all-Korean or empty
            continue

        seg_eng, seg_mand, seg_pin = seg_result

        # Store the info for later output
        text_triplets.append({
            "original_text": original_text,
            "seg_eng": seg_eng,
            "seg_mand": seg_mand,
            "seg_pin": seg_pin,
            "mapping_str": mapping_str,
            "translated_text": translated_text,
            "alignment_info": alignment_info
        })

        # Draw a white rectangle + red outline
        draw.rectangle([(min_x, min_y), (max_x, max_y)], fill=(255,255,255,255))
        draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=(255,0,0,255), width=2)

        box_width = max_x - min_x
        box_height = max_y - min_y

        # Find a font size that fits
        font_size = initial_font_size
        while True:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()

            pinyin_lines, pinyin_height = wrap_tokens(seg_pin, font, box_width)
            english_lines, english_height = wrap_tokens(seg_eng, font, box_width)
            total_text_height = pinyin_height + english_height + SEPARATOR_PADDING

            if total_text_height > box_height and font_size > min_font_size:
                font_size -= 2
            else:
                break

        # Draw the text
        start_y_text = min_y + (box_height - total_text_height) / 2
        start_y_text = draw_wrapped_lines(draw, pinyin_lines, font, min_x, start_y_text, box_width)
        start_y_text += SEPARATOR_PADDING
        # optional horizontal line
        draw.line([(min_x, start_y_text), (min_x + box_width, start_y_text)], fill="black", width=1)
        start_y_text += 1
        draw_wrapped_lines(draw, english_lines, font, min_x, start_y_text, box_width)

    return img.convert("RGB"), text_triplets

# ------------------ STREAMLIT APP ------------------
def main():
    st.title("Batch CBZ Translator with Box Expansion & Word Align Preview")

    # Create a placeholder at the top for a download button (we'll fill it later)
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

    st.write("Upload multiple CBZ files (only for your session).")
    uploaded_files = st.file_uploader("Upload CBZ Files", type=["cbz"], accept_multiple_files=True)

    if uploaded_files:
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        processed_cbz_paths = []

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            cbz_filename = uploaded_file.name
            temp_cbz_path = os.path.join(base_temp_folder, cbz_filename)
            with open(temp_cbz_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract images
            temp_extract_folder = os.path.join(base_temp_folder, f"extract_{uuid.uuid4().hex}")
            os.makedirs(temp_extract_folder, exist_ok=True)
            images = extract_cbz(temp_cbz_path, temp_extract_folder)

            # Process each image
            for img_path in images:
                annotations = detect_text_boxes(img_path)
                if annotations:
                    merged_items = group_annotations(annotations)
                    # Expand bounding boxes
                    merged_items = expand_boxes(merged_items, Image.open(img_path).width, Image.open(img_path).height)

                    # Show a preview of the bounding boxes on the original image
                    preview_img = draw_preview_boxes(img_path, merged_items)
                    st.image(preview_img, caption=f"Bounding Box Preview: {os.path.basename(img_path)}")

                    # Now do the overlay
                    final_img, text_triplets = overlay_merged_pinyin(
                        img_path, merged_items,
                        font_path=FONT_PATH,
                        margin=MARGIN
                    )

                    # Display the final image with overlay
                    st.image(final_img, caption=f"Final Overlay: {os.path.basename(img_path)}")

                    # Print out the recognized text, pinyin, and word mapping
                    for block_idx, data in enumerate(text_triplets, start=1):
                        st.write("---")
                        st.write(f"**Text Block {block_idx}:**")
                        st.write("**Original OCR Text:**")
                        st.write(data["original_text"])

                        st.write("**Chinese (segmented):**")
                        seg_chinese_str = " ".join([tok[0] for tok in data["seg_mand"]])
                        st.write(seg_chinese_str)

                        st.write("**Pinyin (segmented):**")
                        seg_pinyin_str = " ".join([tok[0] for tok in data["seg_pin"]])
                        st.write(seg_pinyin_str)

                        st.write("**Word Alignments (English -> Chinese):**")
                        # data["alignment_info"] is a list of (src_i, tgt_j, prob)
                        seg_eng = data["seg_eng"]
                        seg_mand = data["seg_mand"]
                        for (src_i, tgt_j, prob) in data["alignment_info"]:
                            if src_i < len(seg_eng) and tgt_j < len(seg_mand):
                                eng_word = seg_eng[src_i][0]
                                cn_word = seg_mand[tgt_j][0]
                                st.write(f" - {eng_word} -> {cn_word} (prob={prob:.3f})")

                    # Finally, overwrite the original image on disk with the final overlay
                    final_img.save(img_path)

            # Repack to CBZ
            output_cbz_path = os.path.join(output_folder, cbz_filename)
            repack_to_cbz(temp_extract_folder, output_cbz_path)
            processed_cbz_paths.append(output_cbz_path)

            # Cleanup the extracted folder
            shutil.rmtree(temp_extract_folder)
            progress_bar.progress(idx / total_files)

        st.success("Processing complete!")

        # Create a ZIP of all processed CBZ
        final_zip = io.BytesIO()
        with zipfile.ZipFile(final_zip, "w") as zf:
            for file in sorted(os.listdir(output_folder)):
                if file.lower().endswith(".cbz"):
                    full_path = os.path.join(output_folder, file)
                    zf.write(full_path, arcname=file)

        final_zip.seek(0)

        # Place the download button in our placeholder *at the top*
        download_placeholder.download_button(
            label="Download Processed CBZ Files (ZIP)",
            data=final_zip,
            file_name="processed_cbz_files.zip",
            mime="application/zip"
        )

    # Optional button to reset
    if st.button("Clear My Session Files"):
        shutil.rmtree(base_temp_folder, ignore_errors=True)
        shutil.rmtree(output_folder, ignore_errors=True)
        st.session_state.pop("session_id", None)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
