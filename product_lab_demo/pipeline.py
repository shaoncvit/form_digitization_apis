import os
import cv2
import pandas as pd
import numpy as np
import easyocr
import argparse
import json
from collections import defaultdict
import re

def draw_boxes(image, annotations, output_path):
    img = image.copy()
    for idx, row in annotations.iterrows():
        x, y, w, h = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_width']), int(row['bbox_height'])
        label = row['label_name']
        color = (0, 255, 0) if 'key' in label else (255, 0, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(output_path, img)

def read_annotations(csv_path, image_name):
    print('DEBUG: Reading annotations from', csv_path)
    df = pd.read_csv(csv_path)
    print('DEBUG: Annotation file loaded, shape:', df.shape)
    # df = df[df['image_name'] == image_name]  # Commented out for debugging
    print('DEBUG: Annotation head:', df.head())
    return df

def extract_regions(image, annotations):
    regions = {}
    label_counts = defaultdict(int)
    for idx, row in annotations.iterrows():
        label = row['label_name']
        label_counts[label] += 1
        unique_label = f"{label}_{label_counts[label]}"
        x, y, w, h = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_width']), int(row['bbox_height'])
        crop = image[y:y+h, x:x+w]
        regions[unique_label] = crop
    return regions

def ocr_regions(regions, reader):
    ocr_results = {}
    for label, img in regions.items():
        if img.size == 0:
            ocr_results[label] = ""
            continue
        result = reader.readtext(img, detail=0, paragraph=True)
        print(result)
        # Clean each result before joining
        cleaned = [clean_text(r) for r in result]
        ocr_results[label] = " ".join(cleaned)
    return ocr_results

def clean_text(text):
    # Remove non-printable characters and excessive whitespace
    return re.sub(r'[^\x20-\x7E]+', '', text).strip()

def group_key_values(ocr_results, annotations):
    key_label_to_text = {}
    key_label_to_vals = defaultdict(list)
    label_counts = defaultdict(int)
    # First, get all key texts with their unique indices
    for idx, row in annotations.iterrows():
        label = row['label_name']
        label_counts[label] += 1
        unique_label = f"{label}_{label_counts[label]}"
        if label.startswith('key_'):
            key_label_to_text[label] = ocr_results[unique_label]
    # Reset for values
    label_counts = defaultdict(int)
    for idx, row in annotations.iterrows():
        label = row['label_name']
        label_counts[label] += 1
        unique_label = f"{label}_{label_counts[label]}"
        if label.startswith('val_'):
            key_num = label.split('_')[1]
            key_label = f'key_{key_num}'
            key_label_to_vals[key_label].append(ocr_results[unique_label])
    result_json = {}
    for key_label, key_text in key_label_to_text.items():
        vals = key_label_to_vals.get(key_label, [])
        result_json[key_text] = ", ".join([v for v in vals if v])
    return result_json

def main(image_path, template_name, output_prefix=None):
    print('DEBUG: main() called with')
    print('  image_path:', image_path)
    print('  template_name:', template_name)
    print('  output_prefix:', output_prefix)
    annotation_path = f'annotations/{template_name}.csv'
    print('  annotation_path:', annotation_path)
    image_name = os.path.basename(image_path)
    print('  image_name (for annotation filter):', image_name)
    image = cv2.imread(image_path)
    print('  image is None?', image is None)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return
    annotations = read_annotations(annotation_path, image_name)
    print('  annotations shape:', annotations.shape)
    print('  annotations head:', annotations.head())
    # Use absolute output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    # Use output_prefix for output files
    if output_prefix is None:
        output_prefix = template_name
    boxed_img_path = os.path.join(output_dir, f'{output_prefix}_boxed.jpg')
    draw_boxes(image, annotations, boxed_img_path)
    print(f"Saved boxed image to {boxed_img_path}")
    regions = extract_regions(image, annotations)
    reader = easyocr.Reader(['en'])
    ocr_results = ocr_regions(regions, reader)
    result_json = group_key_values(ocr_results, annotations)
    json_path = os.path.join(output_dir, f'{output_prefix}_ocr.json')
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"Saved OCR results to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--template', type=str, required=True, help='Template name (e.g. hdfc_bank)')
    parser.add_argument('--output_prefix', type=str, default=None, help='Prefix for output files')
    args = parser.parse_args()
    main(args.image, args.template, args.output_prefix)