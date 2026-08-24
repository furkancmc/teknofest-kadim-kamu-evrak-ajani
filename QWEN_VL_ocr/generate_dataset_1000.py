import os
import re
import sys
import math
import random
import io
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datasets import load_from_disk
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# PyMuPDF for PDF rendering
import pymupdf as fitz

# WeasyPrint fallback check
WEASYPRINT_AVAILABLE = False
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False


def clean_markdown(text: str) -> str:
    """
    Cleans markdown syntax (**bold**, *italic*, # headers, etc.) from text,
    returning clean plain text while preserving Turkish characters and line breaks.
    """
    if not text:
        return ""
    
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def select_1000_samples(dataset, seed=42):
    """
    Selects 1000 diverse samples from dataset and partitions them into:
    - Train: 800 (200 A, 200 B, 200 C, 200 D)
    - Valid: 100 ( 25 A,  25 B,  25 C,  25 D)
    - Test:  100 ( 25 A,  25 B,  25 C,  25 D)
    Total per group = 250 samples.
    """
    random.seed(seed)
    
    split_targets = {
        'A': {'train': 200, 'valid': 25, 'test': 25},
        'B': {'train': 200, 'valid': 25, 'test': 25},
        'C': {'train': 200, 'valid': 25, 'test': 25},
        'D': {'train': 200, 'valid': 25, 'test': 25},
    }
    
    total_needed = 1000
    total_len = len(dataset)
    
    # Fast candidate scanning pool
    indices = list(range(total_len))
    random.shuffle(indices)
    sample_pool_size = min(15000, total_len)
    sample_indices = indices[:sample_pool_size]
    
    print(f"Fast-scanning candidate pool of {sample_pool_size} records (from {total_len} total)...")
    
    candidates = []
    for idx in sample_indices:
        record = dataset[idx]
        raw_text = record.get('text', '')
        cleaned = clean_markdown(raw_text)
        length = len(cleaned)
        
        category = None
        if 500 <= length < 1000:
            category = 'short'
        elif 1000 <= length < 3000:
            category = 'medium'
        elif 3000 <= length <= 6000:
            category = 'long'
            
        if category:
            candidates.append({
                'index': idx,
                'esasNo': record.get('esasNo', ''),
                'kararNo': record.get('kararNo', ''),
                'kararTarihi': record.get('kararTarihi', ''),
                'source': str(record.get('source', '')),
                'labels': str(record.get('labels', '')),
                'cleaned_text': cleaned,
                'category': category
            })
            
    print(f"Candidate pool categorized: {len(candidates)} valid items found.")
    
    if len(candidates) < total_needed:
        print(f"Warning: Only {len(candidates)} candidates found, needed {total_needed}.")
        
    random.shuffle(candidates)
    selected_1000 = candidates[:total_needed]
    
    # Assign groups (250 A, 250 B, 250 C, 250 D)
    groups = ['A'] * 250 + ['B'] * 250 + ['C'] * 250 + ['D'] * 250
    random.shuffle(groups)
    
    group_items = {'A': [], 'B': [], 'C': [], 'D': []}
    for item, group in zip(selected_1000, groups):
        item['group'] = group
        group_items[group].append(item)
        
    all_final_items = []
    
    # Partition each group into train, valid, test
    for group_code in ['A', 'B', 'C', 'D']:
        items_in_g = group_items[group_code]
        t_train = split_targets[group_code]['train']
        t_val = split_targets[group_code]['valid']
        t_test = split_targets[group_code]['test']
        
        train_part = items_in_g[:t_train]
        val_part = items_in_g[t_train:t_train + t_val]
        test_part = items_in_g[t_train + t_val:t_train + t_val + t_test]
        
        for idx, item in enumerate(train_part, 1):
            item['split'] = 'train'
            item['id'] = f"{group_code}_{idx:04d}"
            all_final_items.append(item)
            
        for idx, item in enumerate(val_part, 1):
            item['split'] = 'valid'
            item['id'] = f"{group_code}_{idx:04d}"
            all_final_items.append(item)
            
        for idx, item in enumerate(test_part, 1):
            item['split'] = 'test'
            item['id'] = f"{group_code}_{idx:04d}"
            all_final_items.append(item)
            
    print(f"Selected total {len(all_final_items)} items assigned to Train/Valid/Test splits.")
    return all_final_items


def get_html_template(item, style_idx=0):
    """
    Generates HTML representation for a document with antet header & clean text body.
    Supports 4 distinct visual layout styles and font families.
    """
    source_lower = item['source'].lower()
    if 'dns' in source_lower or 'danıstay' in source_lower:
        antet_title = "T.C. DANIŞTAY KARARI"
    else:
        antet_title = "T.C. YARGITAY KARARI"
        
    esas_no = item['esasNo'] or "2021/1042"
    karar_no = item['kararNo'] or "2022/4510"
    karar_tarihi = item['kararTarihi'] or "15.03.2022"
    belge_id = item['id']
    
    paragraphs = item['cleaned_text'].split('\n\n')
    body_html = "".join([f"<p>{p.replace('\n', '<br>')}</p>" for p in paragraphs if p.strip()])
    
    variant = style_idx % 4
    
    if variant == 0:
        font_name = "Times New Roman"
        style_name = "Classic Court (Serif)"
        css = """
        @page { size: A4; margin: 20mm; }
        body { font-family: 'Times New Roman', Georgia, serif; font-size: 12pt; line-height: 1.45; color: #111; }
        .header { text-align: center; border-bottom: 3px double #333; padding-bottom: 12px; margin-bottom: 20px; }
        .header h1 { font-size: 16pt; font-weight: bold; margin: 0 0 8px 0; letter-spacing: 1px; }
        .meta-table { width: 100%; font-size: 10.5pt; font-weight: bold; margin-top: 6px; }
        .meta-table td { padding: 2px 4px; }
        .content { text-align: justify; text-justify: inter-word; text-indent: 2em; }
        .content p { margin-bottom: 12px; }
        """
    elif variant == 1:
        font_name = "Arial"
        style_name = "Modern Official (Sans-Serif)"
        css = """
        @page { size: A4; margin: 18mm; }
        body { font-family: Arial, 'Helvetica Neue', sans-serif; font-size: 11pt; line-height: 1.4; color: #1a1a1a; }
        .header { background-color: #f4f4f6; border-left: 4px solid #2b4c7e; padding: 12px 16px; margin-bottom: 22px; }
        .header h1 { font-size: 15pt; font-weight: bold; color: #2b4c7e; margin: 0 0 6px 0; }
        .meta-table { width: 100%; font-size: 10pt; color: #444; }
        .meta-table td { padding: 2px 0; }
        .content { text-align: justify; text-indent: 1.5em; }
        .content p { margin-bottom: 10px; }
        """
    elif variant == 2:
        font_name = "Bookman Old Style"
        style_name = "Structured Boxed (Bookman)"
        css = """
        @page { size: A4; margin: 22mm; }
        body { font-family: 'Bookman Old Style', Georgia, serif; font-size: 11.5pt; line-height: 1.38; color: #000; }
        .header { border: 1px solid #444; padding: 12px; margin-bottom: 20px; text-align: center; }
        .header h1 { font-size: 15pt; text-transform: uppercase; margin: 0 0 10px 0; border-bottom: 1px solid #aaa; padding-bottom: 6px; }
        .meta-table { width: 100%; font-size: 10pt; text-align: left; }
        .meta-table td { padding: 3px 6px; }
        .content { text-align: justify; text-indent: 2em; }
        .content p { margin-bottom: 14px; }
        """
    else:
        font_name = "Courier New"
        style_name = "Typewriter Style (Monospace)"
        css = """
        @page { size: A4; margin: 20mm; }
        body { font-family: 'Courier New', Courier, monospace; font-size: 10.5pt; line-height: 1.35; color: #050505; }
        .header { text-align: center; border-bottom: 1px dashed #222; padding-bottom: 10px; margin-bottom: 18px; }
        .header h1 { font-size: 14pt; font-weight: bold; margin: 0 0 6px 0; letter-spacing: 0.5px; }
        .meta-table { width: 100%; font-size: 10pt; margin-top: 4px; }
        .meta-table td { padding: 2px 2px; }
        .content { text-align: justify; text-indent: 2em; }
        .content p { margin-bottom: 10px; }
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{css}
</style>
</head>
<body>
<div class="header">
    <h1>{antet_title}</h1>
    <table class="meta-table">
        <tr>
            <td><strong>Esas No:</strong> {esas_no}</td>
            <td><strong>Karar No:</strong> {karar_no}</td>
        </tr>
        <tr>
            <td><strong>Karar Tarihi:</strong> {karar_tarihi}</td>
            <td><strong>Belge ID:</strong> {belge_id}</td>
        </tr>
    </table>
</div>
<div class="content">
{body_html}
</div>
</body>
</html>
"""
    return html, font_name, style_name


def render_html_to_pil_image(html_content: str, dpi: int = 150) -> Image.Image:
    """
    Renders HTML string to PDF and converts PDF page(s) into a single PIL Image.
    """
    pdf_bytes = None
    if WEASYPRINT_AVAILABLE:
        try:
            pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        except Exception:
            pdf_bytes = None
            
    if pdf_bytes is None:
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        rect = fitz.Rect(0, 0, 595, 842)
        try:
            page.insert_htmlbox(rect, html_content)
        except AttributeError:
            clean_txt = re.sub(r'<[^>]+>', ' ', html_content)
            page.insert_textbox(rect, clean_txt[:3000], fontsize=11)
        pdf_bytes = doc.tobytes()
        doc.close()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    page_images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(img)
    doc.close()
    
    if len(page_images) == 1:
        return page_images[0]
    else:
        total_height = sum(img.height for img in page_images)
        max_width = max(img.width for img in page_images)
        combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
        y_offset = 0
        for img in page_images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height
        return combined


def apply_augmentations(img: Image.Image, group: str, dpi: int):
    """
    Applies image augmentations following real-world scanner/camera processing order.
    """
    meta = {
        'group': group,
        'dpi': dpi,
        'blur_sigma': 0.0,
        'contrast_factor': 1.0,
        'brightness_factor': 1.0,
        'gaussian_noise_sigma': 0.0,
        'salt_pepper_ratio': 0.0,
        'rotation_deg': 0.0,
        'jpeg_quality': None
    }
    
    if group == 'A':
        return img, meta

    # STEP 1: BLUR
    if group == 'B':
        sigma = random.uniform(0.5, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        meta['blur_sigma'] = round(sigma, 2)
    elif group == 'C':
        sigma = random.uniform(1.2, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        meta['blur_sigma'] = round(sigma, 2)
    elif group == 'D':
        sigma = random.uniform(1.5, 2.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        meta['blur_sigma'] = round(sigma, 2)

    # STEP 2: BRIGHTNESS & CONTRAST
    if group == 'B':
        contrast = random.uniform(0.85, 1.05)
        brightness = random.uniform(0.90, 1.10)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        meta['contrast_factor'] = round(contrast, 3)
        meta['brightness_factor'] = round(brightness, 3)
    elif group == 'C':
        contrast = random.uniform(0.75, 0.95)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        meta['contrast_factor'] = round(contrast, 3)
    elif group == 'D':
        contrast = random.uniform(0.70, 0.90)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        meta['contrast_factor'] = round(contrast, 3)

    # STEP 3: NOISE
    if group in ('C', 'D'):
        img_np = np.array(img).astype(np.float32)
        if group == 'C':
            g_sigma = random.uniform(5.0, 10.0)
            noise = np.random.normal(0, g_sigma, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255)
            meta['gaussian_noise_sigma'] = round(g_sigma, 1)
        elif group == 'D':
            g_sigma = random.uniform(8.0, 15.0)
            noise = np.random.normal(0, g_sigma, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255)
            meta['gaussian_noise_sigma'] = round(g_sigma, 1)
            
            sp_ratio = random.uniform(0.002, 0.006)
            num_sp = int(sp_ratio * img_np.shape[0] * img_np.shape[1])
            ys = np.random.randint(0, img_np.shape[0], num_sp // 2)
            xs = np.random.randint(0, img_np.shape[1], num_sp // 2)
            img_np[ys, xs] = 255
            yp = np.random.randint(0, img_np.shape[0], num_sp // 2)
            xp = np.random.randint(0, img_np.shape[1], num_sp // 2)
            img_np[yp, xp] = 0
            meta['salt_pepper_ratio'] = round(sp_ratio, 4)
            
        img = Image.fromarray(img_np.astype(np.uint8))

    # STEP 4: ROTATION
    if group in ('C', 'D'):
        max_angle = 3.0 if group == 'C' else 4.0
        angle = random.uniform(-max_angle, max_angle)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        meta['rotation_deg'] = round(angle, 2)

    # STEP 5: JPEG COMPRESSION
    if group in ('B', 'D'):
        q_range = (85, 95) if group == 'B' else (50, 75)
        quality = random.randint(q_range[0], q_range[1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        meta['jpeg_quality'] = quality

    return img, meta


def process_single_item(args):
    """
    Worker function to render document into dataset_1000/<split>/ directory.
    """
    item, base_output_dir = args
    item_id = item['id']
    group = item['group']
    split = item['split']
    
    split_dir = os.path.join(base_output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    try:
        num_part = int(item_id.split('_')[-1])
    except Exception:
        num_part = 0
    style_idx = num_part % 4
    
    dpi_map = {'A': 200, 'B': 200, 'C': 150, 'D': 120}
    target_dpi = dpi_map.get(group, 150)
    
    html_content, font_name, style_name = get_html_template(item, style_idx=style_idx)
    raw_img = render_html_to_pil_image(html_content, dpi=target_dpi)
    
    final_img, meta_dict = apply_augmentations(raw_img, group, dpi=target_dpi)
    meta_dict['id'] = item_id
    meta_dict['split'] = split
    meta_dict['font_name'] = font_name
    meta_dict['style_name'] = style_name
    meta_dict['source'] = item['source']
    meta_dict['length_cat'] = item['category']
    meta_dict['char_len'] = len(item['cleaned_text'])
    
    png_path = os.path.join(split_dir, f"{item_id}.png")
    txt_path = os.path.join(split_dir, f"{item_id}.txt")
    
    final_img.save(png_path, format="PNG")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(item['cleaned_text'])
        
    return {
        'id': item_id,
        'group': group,
        'split': split,
        'length_cat': item['category'],
        'char_len': len(item['cleaned_text']),
        'source': item['source'],
        'png_path': png_path,
        'txt_path': txt_path,
        'meta': meta_dict
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 1000-sample Turkish Law OCR dataset with Train/Valid/Test splits.")
    parser.add_argument("--dataset_path", type=str, default="./turkish_law_dataset", help="Path to saved disk dataset.")
    parser.add_argument("--output_dir", type=str, default="./dataset_1000", help="Base output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing (0 = full 1000 dataset).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from '{args.dataset_path}'...")
    dataset = load_from_disk(args.dataset_path)
    if hasattr(dataset, 'keys') and 'train' in dataset.keys():
        print("Extracting 'train' split from DatasetDict...")
        dataset = dataset['train']
        
    print("Selecting 1000 stratified records...")
    selected_items = select_1000_samples(dataset, seed=args.seed)
    
    if args.limit > 0:
        selected_items = selected_items[:args.limit]
        print(f"Limit applied: Processing first {len(selected_items)} items...")
        
    engine_name = "WeasyPrint (Full CSS Engine)" if WEASYPRINT_AVAILABLE else "PyMuPDF (Fallback Native Engine)"
    print(f"\nPDF Rendering Engine: {engine_name}")
    print(f"Starting parallel rendering for {len(selected_items)} items into '{args.output_dir}'...\n")
    
    results = []
    metadata_logs = {}
    
    tasks = [(item, args.output_dir) for item in selected_items]
    
    completed_count = 0
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        for res in executor.map(process_single_item, tasks):
            results.append(res)
            metadata_logs[res['id']] = res['meta']
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == len(selected_items):
                print(f"  [+] Progress: {completed_count}/{len(selected_items)} items rendered...")
                
    log_file_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata_logs, f, ensure_ascii=False, indent=2)
    print(f"\nSaved complete dataset metadata log to: {log_file_path}")

    # Calculate Breakdown Matrix
    matrix = {
        'train': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
        'valid': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
        'test':  {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    }
    
    for r in results:
        matrix[r['split']][r['group']] += 1
        
    print("\n" + "="*60)
    print(" 1,000 DATASET GENERATION SUMMARY MATRIX ")
    print("="*60)
    print(f"{'Group':<8} | {'Train':<8} | {'Valid':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 60)
    for g in ['A', 'B', 'C', 'D']:
        tr = matrix['train'][g]
        va = matrix['valid'][g]
        te = matrix['test'][g]
        tot = tr + va + te
        print(f"Group {g:<3} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")
    print("-" * 60)
    tot_tr = sum(matrix['train'].values())
    tot_va = sum(matrix['valid'].values())
    tot_te = sum(matrix['test'].values())
    print(f"{'TOTAL':<8} | {tot_tr:<8} | {tot_va:<8} | {tot_te:<8} | {tot_tr + tot_va + tot_te:<8}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
