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


st.write("Checking internet...")
try:
    import requests
    st.write(requests.get("https://huggingface.co").status_code)
except Exception as e:
    st.write("Internet error:", e)

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



def detect_blocks(image_path):
    with open(image_path, "rb") as img_file:
        content = img_file.read()
    image = vision.Image(content=content)
    response = vision_client.document_text_detection(image=image)
    annotation = response.full_text_annotation

    if not annotation or not annotation.pages:
        return []

    blocks = []
    for page in annotation.pages:
        for block in page.blocks:
            # Reconstruct block text by concatenating all paragraphs in the block
            block_text = ""
            for paragraph in block.paragraphs:
                paragraph_text = " ".join("".join(symbol.text for symbol in word.symbols)
                                          for word in paragraph.words)
                block_text += paragraph_text + " "
            block_text = block_text.strip()

            # --- NEW: Skip blocks with less than 3 words ---
            if len(block_text.split()) < 3:
                continue

            # Convert the block's bounding_poly vertices into a simple box
            vertices = [(v.x, v.y) for v in block.bounding_box.vertices]
            xs, ys = zip(*vertices)
            bbox = (min(xs), min(ys), max(xs), max(ys))

            blocks.append({
                "bbox": bbox,
                "text": block_text
            })
    return blocks


def combine_close_blocks(blocks, threshold=10):
    def overlap_or_close(boxA, boxB, threshold=10):
        Aminx, Aminy, Amaxx, Amaxy = boxA
        Bminx, Bminy, Bmaxx, Bmaxy = boxB
        if Amaxx < (Bminx - threshold) or Bmaxx < (Aminx - threshold):
            return False
        if Amaxy < (Bminy - threshold) or Bmaxy < (Aminy - threshold):
            return False
        return True

    def merge_boxes_and_text(boxA, boxB, textA, textB):
        Aminx, Aminy, Amaxx, Amaxy = boxA
        Bminx, Bminy, Bmaxx, Bmaxy = boxB
        merged_box = (
            min(Aminx, Bminx),
            min(Aminy, Bminy),
            max(Amaxx, Bmaxx),
            max(Amaxy, Bmaxy)
        )
        merged_text = textA + " " + textB  # combine with a space
        return merged_box, merged_text

    # Sort top-to-bottom, then left-to-right
    items = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    merged = True
    while merged:
        merged = False
        new_items = []
        while items:
            current = items.pop()
            for idx, existing in enumerate(new_items):
                if overlap_or_close(current["bbox"], existing["bbox"], threshold=threshold):
                    merged_box, merged_text = merge_boxes_and_text(
                        current["bbox"], existing["bbox"],
                        current["text"], existing["text"]
                    )
                    new_items[idx]["bbox"] = merged_box
                    new_items[idx]["text"] = merged_text
                    merged = True
                    break
            else:
                new_items.append(current)
        items = new_items

    # === Final hyphenation cleanup on each block's text ===
    for item in items:
        item["text"] = remove_hyphenation(item["text"])

    return items



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
    # text_annotations[0] is entire text, skip that
    return response.text_annotations[1:]

# ------------------ AWESOME-ALIGN MODEL ------------------
@st.cache_resource
def load_awesome_align_model():
    model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co",
                                     use_auth_token = "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")
    tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co",
                                             use_auth_token = "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")
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
    3) Concatenate the sentence-level tokens into a single set.
       Return (segmented_eng, segmented_mand, segmented_pin), combined alignment placeholder, combined Chinese text.
    """

    # Break up into naive sentences
    sentence_list = split_into_sentences(english_text)

    # We'll accumulate tokens for all sentences
    all_seg_eng = []
    all_seg_mand = []
    all_seg_pin = []

    normal_palette = [
        "blue", "green", "orange", "purple", "brown",
        "cyan", "magenta", "olive", "teal", "navy"
    ]
    color_idx = 0  # which color in the palette to use next

    for sent in sentence_list:
        # If it's purely Korean, skip alignment
        if is_all_korean(sent):
            continue

        # Filter out tokens that contain ANY Korean
        filtered_text = filter_korean(sent)
        if not filtered_text.strip():
            continue

        # Translate leftover text to Chinese
        cn_text = translate_text(filtered_text)

        # Run Awesome-Align
        mapping_str = awesome_align(filtered_text, cn_text)
        st.write("Awesome-Align mapping (per sentence):", mapping_str)

        # Parse the alignment string
        try:
            mapping_set = eval(mapping_str)
            mapping_pairs = [tup for tup in mapping_set if len(tup) == 3]
        except Exception as e:
            st.write("Error parsing mapping_str:", e)
            mapping_pairs = []

        mapping_dict = {}
        reverse_mapping = {}
        for i, j, _ in mapping_pairs:
            mapping_dict.setdefault(i, set()).add(j)
            reverse_mapping.setdefault(j, set()).add(i)

        # Tokenize English and Chinese
        sent_src = filtered_text.strip().split()
        sent_tgt = list(jieba.cut(cn_text))

        # Assign colors within this sentence
        color_mapping = {}
        for i, word in enumerate(sent_src):
            if i in mapping_dict and color_idx < len(normal_palette):
                color_mapping[i] = normal_palette[color_idx]
                color_idx = (color_idx + 1) % len(normal_palette)
            else:
                color_mapping[i] = "black"

        target_color_mapping = {}
        for j, word in enumerate(sent_tgt):
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

        # Add space tokens between sentence chunks (optional)
        if all_seg_eng:
            all_seg_eng.append((" ", "black"))
            all_seg_mand.append((" ", "black"))
            all_seg_pin.append((" ", "black"))

        # Append tokens for this sentence
        all_seg_eng.extend(seg_eng)
        all_seg_mand.extend(seg_mand)
        all_seg_pin.extend(seg_pin)

    # If we never got any tokens, it means the entire text was Korean or filtered out
    if not all_seg_eng:
        return None, "", english_text

    # Combine final Chinese text for display (just concatenates the Chinese tokens)
    combined_chinese = " ".join([tok[0] for tok in all_seg_mand])
    # We won't store a real alignment string for the entire multi-sentence text, 
    # but we can put a placeholder
    mapping_str = "<multi-sentence alignment>"

    return (all_seg_eng, all_seg_mand, all_seg_pin), mapping_str, combined_chinese

# ------------------ MERGING LOGIC ------------------
def bbox_for_annotation(ann):
    vs = ann.bounding_poly.vertices
    xs = [v.x for v in vs]
    ys = [v.y for v in vs]
    return min(xs), min(ys), max(xs), max(ys)

def overlap_or_close(boxA, boxB, threshold=MERGE_THRESHOLD):
    """
    Merge logic uses a small threshold to see if they should combine as one text region.
    """
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

def remove_hyphenation(text):
    """
    Attempt to fix common hyphenation issues from OCR, e.g.:
      "IMPOR- TANT" => "IMPORTANT"
      "INFOR- MATION." => "INFORMATION."
    """
    # 1) Replace any newlines with a space
    text = text.replace("\n", " ")
    
    # 2) Convert em dashes or en dashes to a normal dash
    text = text.replace("—", "-").replace("–", "-")
    
    # 3) Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    
    # 4) Merge patterns like "word- word" => "wordword"
    text = re.sub(r"(\S+)-\s+(\S+)", r"\1\2", text)
    
    return text



def group_annotations(annotations):
    """
    Merges overlapping text boxes into single items with combined text.
    Now sorts bounding boxes top-to-bottom, left-to-right before merging
    so that final text is in a more natural reading order.
    """
    # 1. Convert each annotation into (bbox, text) but do NOT remove hyphenation yet.
    items = []
    for ann in annotations:
        raw_text = ann.description
        box = bbox_for_annotation(ann)
        items.append({"bbox": box, "text": raw_text})
    
    # 2. Sort items by their top (min_y), then by left (min_x).
    #    This should help preserve reading order when we merge text.
    def sort_key(item):
        x1, y1, x2, y2 = item["bbox"]
        return (y1, x1)  # sort by top first, then left
    items.sort(key=sort_key)
    
    # 3. Merge pass
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

    # 4. After merging is done, do final hyphenation cleanup on each merged text.
    for it in items:
        it["text"] = remove_hyphenation(it["text"])
    
    return items



# ------------------ HELPER: BOX OVERLAP WITHOUT THRESHOLD ------------------
def boxes_overlap(boxA, boxB):
    """
    Simple bounding-box overlap check with zero threshold.
    """
    Aminx, Aminy, Amaxx, Amaxy = boxA
    Bminx, Bminy, Bmaxx, Bmaxy = boxB
    if Amaxx < Bminx or Bmaxx < Aminx:
        return False
    if Amaxy < Bminy or Bmaxy < Aminy:
        return False
    return True

# ------------------ HELPER: TRY EXPANDING A BOX ------------------
def try_expand_box(orig_box, other_boxes, img_width, img_height,
                   expand_w_factor=0.2, expand_h_factor=0.35,  # <-- CHANGED HERE
                   margin=5, extra_padding=10):
    """
    Attempt to expand orig_box by 20% in width, *35%* in height
    (was 25% originally), plus existing margin/padding.
    If expansion causes overlap with any box in other_boxes, revert to original.

    Returns the final (min_x, min_y, max_x, max_y).
    """
    (min_x, min_y, max_x, max_y) = orig_box

    # Original expansions (margin + padding)
    min_x = max(0, min_x - margin - extra_padding)
    min_y = max(0, min_y - margin - extra_padding)
    max_x = min(img_width, max_x + margin + extra_padding)
    max_y = min(img_height, max_y + margin + extra_padding)

    orig_expanded = (min_x, min_y, max_x, max_y)

    # Now compute the expand_w_factor & expand_h_factor expansions around the center
    width = max_x - min_x
    height = max_y - min_y
    expand_w = width * expand_w_factor
    expand_h = height * expand_h_factor

    new_min_x = min_x - expand_w / 2
    new_max_x = max_x + expand_w / 2
    new_min_y = min_y - expand_h / 2
    new_max_y = max_y + expand_h / 2

    # Clamp to image boundaries
    new_min_x = max(0, new_min_x)
    new_min_y = max(0, new_min_y)
    new_max_x = min(img_width, new_max_x)
    new_max_y = min(img_height, new_max_y)

    expanded_box = (new_min_x, new_min_y, new_max_x, new_max_y)

    # Check overlap with others (excluding the original box itself).
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

# ------------------ OVERLAY ------------------
def overlay_merged_pinyin(image_path, items, font_path=FONT_PATH, margin=MARGIN):
    """
    For each annotation box, we do:
      1) Attempt to expand the bounding box by 20% width, 35% height 
         (unless it overlaps).
      2) If text is all Korean, skip overlay.
      3) Otherwise, translate & overlay pinyin on top, separator, then English.
    """
    EXTRA_PADDING = 10
    SEPARATOR_PADDING = 10
    initial_font_size = 22
    min_font_size = 14

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    text_triplets = []

    # === First pass: expand each bounding box if possible ===
    for item in items:
        item["final_bbox"] = None

    for i, item in enumerate(items):
        orig_box = item["bbox"]
        expanded_box = try_expand_box(
            orig_box,
            other_boxes=[it for it in items if it is not item],
            img_width=img.width,
            img_height=img.height,
            expand_w_factor=0.2,
            expand_h_factor=0.35,  # vertical expansion increased
            margin=margin,
            extra_padding=EXTRA_PADDING
        )
        item["final_bbox"] = expanded_box

    # === Second pass: do the actual overlay with the final bounding box. ===
    for item in items:
        (min_x, min_y, max_x, max_y) = item["final_bbox"]
        original_text = item["text"].strip()

        seg_result, mapping_str, translated_text = translate_to_segments(original_text)
        if seg_result is None:
            continue  # all-Korean or empty after filtering

        seg_eng, seg_mand, seg_pin = seg_result
        text_triplets.append((original_text, (seg_eng, seg_mand, seg_pin), mapping_str, translated_text))

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

        # Draw text
        start_y_text = min_y + (box_height - total_text_height) / 2
        start_y_text = draw_wrapped_lines(draw, pinyin_lines, font, min_x, start_y_text, box_width)

        start_y_text += SEPARATOR_PADDING
        draw.line([(min_x, start_y_text), (min_x + box_width, start_y_text)], fill="black", width=1)
        start_y_text += 1

        draw_wrapped_lines(draw, english_lines, font, min_x, start_y_text, box_width)

    return img.convert("RGB"), text_triplets




def debug_print_ocr_details(image_path, orig_img=None):
    """
    Debug function to:
      - Display the original (before) image with red boxes drawn around the detected block regions.
      - Display the processed (after) image with overlayed translations and pinyin.
      - Print out the extracted block text, the generated pinyin, and the word mapping from Awesome-Align.
      
    Parameters:
      image_path: The path to the image file.
      orig_img: (Optional) A PIL Image object representing the original image before processing.
                If None, the function will load the image from image_path.
    """
    from PIL import Image, ImageDraw
    import streamlit as st

    # Use the provided original image if available; otherwise load from disk.
    if orig_img is None:
        orig_img = Image.open(image_path).convert("RGB")
    else:
        orig_img = orig_img.copy()

    # Create a "before" image copy to draw the block boxes.
    before_img = orig_img.copy()
    draw_before = ImageDraw.Draw(before_img)
    
    # Get block items using your new detect_blocks function.
    block_items = detect_blocks(image_path)
    merged_blocks = combine_close_blocks(block_items, threshold=10)

    if not block_items:
        st.write("No blocks detected in the image.")
        return
    
    # Draw red boxes for each detected block.
    for item in merged_blocks:
        bbox = item["bbox"]
        draw_before.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=2)
        # Optionally, you can annotate with a snippet of block text:
        draw_before.text((bbox[0], bbox[1]-10), item["text"][:30] + "...", fill="red")
    
    # Process the image to get the final overlay image and debug information.
    after_img, text_triplets = overlay_merged_pinyin(image_path, merged_blocks, font_path=FONT_PATH, margin=MARGIN)
    
    # Display the before and after images using Streamlit.
    st.write("**Before Image (Original with red block boxes):**")
    st.image(before_img)
    st.write("**After Image (With overlayed translations and pinyin):**")
    st.image(after_img)
    
    # Print debug info for each block.
    for idx, triplet in enumerate(text_triplets, start=1):
        original_text, (seg_eng, seg_mand, seg_pin), mapping_str, translated_text = triplet
        st.write(f"--- Debug Info for Block {idx} ---")
        st.write("**Extracted Block Text:**", original_text)
        # Combine pinyin tokens into a single string.
        pinyin_text = " ".join([token for token, color in seg_pin])
        st.write("**Pinyin:**", pinyin_text)
        st.write("**Word Mapping:**", mapping_str)



# ------------------ STREAMLIT APP ------------------
def main():
    st.title("Batch CBZ Translator with Box Expansion")
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



    # --- AUDIO PLAYBACK TO KEEP TAB ACTIVE ---
    audio_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        with open("1-hour-and-20-minutes-of-silence.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_placeholder.audio(audio_bytes, format="audio/mp3")
        status_placeholder.info("Audio is playing...")
    except Exception as e:
        status_placeholder.warning("Audio file not found. Audio playback skipped.")
        
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

            # Extract images from CBZ
            temp_extract_folder = os.path.join(base_temp_folder, f"extract_{uuid.uuid4().hex}")
            os.makedirs(temp_extract_folder, exist_ok=True)
            images = extract_cbz(temp_cbz_path, temp_extract_folder)

            # Process each image
            for img_path in images:
                # Load original image for debugging
                original_image = Image.open(img_path).convert("RGB")
                #debug_print_ocr_details(img_path, orig_img=original_image)

                # Detect block-level items and merge close blocks (with hyphen cleanup)
                block_items = detect_blocks(img_path)
                if block_items:
                    merged_blocks = combine_close_blocks(block_items, threshold=10)
                    final_img, _ = overlay_merged_pinyin(
                        img_path,
                        merged_blocks,
                        font_path=FONT_PATH,
                        margin=MARGIN
                    )
                    final_img.save(img_path)

            # Repack processed images into a new CBZ
            output_cbz_path = os.path.join(output_folder, cbz_filename)
            repack_to_cbz(temp_extract_folder, output_cbz_path)
            processed_cbz_paths.append(output_cbz_path)

            # Cleanup extraction folder for current CBZ
            shutil.rmtree(temp_extract_folder)
            progress_bar.progress(idx / total_files)

        st.success("Processing complete!")

        # Create ZIP of processed CBZ files
        final_zip = io.BytesIO()
        with zipfile.ZipFile(final_zip, "w") as zf:
            for file in sorted(os.listdir(output_folder)):
                if file.lower().endswith(".cbz"):
                    full_path = os.path.join(output_folder, file)
                    zf.write(full_path, arcname=file)
        final_zip.seek(0)

        # Trigger browser notification via JavaScript
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

        # Download button for final ZIP
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

if __name__ == "__main__":
    main()
