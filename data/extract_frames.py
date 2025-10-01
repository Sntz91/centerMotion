import cv2
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

LABEL_DICT = {
    'Car': 0,
    'Pedestrian': 1,
    'Bus': 2,
    'Bicycle': 3,
    'Motorcycle': 4,
    'Trailer': 5
}

def extract_frames(video_path, output_dir, prefix="frame"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as JPEG (can also use PNG)
        frame_path = os.path.join(output_dir, f"{prefix}_{i:05d}.jpg")
        cv2.imwrite(frame_path, frame)
    cap.release()
    print(f"Extracted {total_frames} frames to {output_dir}")


def save_per_frame_annotations(xml_path, output_dir, prefix="frame", img_width=1, img_height=1):
    os.makedirs(output_dir, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame_annots = {}
    frame_annots_boxes = {}
    for track in root.findall('.//track'):
        label = track.get('label')
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            if box.get('outside') == '1' or box.get('occluded') == '1':
                continue
            xtl = float(box.get('xtl')) / img_width
            ytl = float(box.get('ytl')) / img_height
            xbr = float(box.get('xbr')) / img_width
            ybr = float(box.get('ybr')) / img_height
            width = xbr - xtl
            height = ybr - ytl

            center_x = (xtl + xbr) / 2 
            center_y = (ytl + ybr) / 2 

            if frame not in frame_annots:
                frame_annots[frame] = []
            if frame not in frame_annots_boxes:
                frame_annots_boxes[frame] = []
            frame_annots[frame].append([center_x, center_y])
            # frame_annots_boxes[frame].append([LABEL_DICT[label], xtl, ytl, xbr, ybr])
            frame_annots_boxes[frame].append([LABEL_DICT[label], center_x, center_y, width, height])

    for frame_idx, centers in frame_annots.items():
        txt_path = os.path.join(output_dir, f"{prefix}_{frame_idx:05d}_points.txt")
        with open(txt_path, "w") as f:
            for center_x, center_y in centers:
                f.write(f"{center_x} {center_y}\n")
    print(f"Saved annotations as .txt for {len(frame_annots)} frames to {output_dir}")

    for frame_idx, boxes in frame_annots_boxes.items():
        txt_path = os.path.join(output_dir, f"{prefix}_{frame_idx:05d}.txt")
        with open(txt_path, "w") as f:
            for label, cx, cy, w, h in boxes:
                f.write(f"{label} {cx} {cy} {w} {h}\n")
    print(f"Saved box annotations as .txt for {len(frame_annots)} frames to {output_dir}")


def temporal_split_folder(image_dir, label_dir, train_ratio=0.8, prefix="frame"):
    frame_files = sorted([f for f in Path(image_dir).glob("*.jpg")])

    total = len(frame_files)
    train_cutoff = int(total * train_ratio)

    train_image_dir = Path(f'{image_dir}/train')
    train_label_dir = Path(f'{label_dir}/train')
    val_image_dir = Path(f'{image_dir}/val')
    val_label_dir = Path(f'{label_dir}/val')

    # train_dir = inputs_dir / "train"
    # val_dir = inputs_dir / "val"
    # train_dir.mkdir(exist_ok=True)
    # val_dir.mkdir(exist_ok=True)
    train_image_dir.mkdir(exist_ok=True)
    train_label_dir.mkdir(exist_ok=True)
    val_image_dir.mkdir(exist_ok=True)
    val_label_dir.mkdir(exist_ok=True)

    for i, frame_file in enumerate(frame_files):
        train_val = 'train' if i < train_cutoff else 'val'
        shutil.move(str(frame_file), f'{image_dir}/{train_val}/{frame_file.name}')

        # Move corresponding annotation .txt if it exists
        annot_file = Path(label_dir) / f"{frame_file.stem}_points.txt"
        if annot_file.exists():
            shutil.move(str(annot_file), f'{label_dir}/{train_val}/{annot_file.name}')

        # Move corresponding annotation .txt if it exists
        annot_file = Path(label_dir) / f"{frame_file.stem}.txt"
        if annot_file.exists():
            shutil.move(str(annot_file), f'{label_dir}/{train_val}/{annot_file.name}')

    print(f"Moved {train_cutoff} frames to 'train/', {total - train_cutoff} to 'val/'")

if __name__ == '__main__':
    video_path = "/home/tobias/data/gta2tv/processed/scenario_1/synced_camera_1.mp4"
    annotation_path = "/home/tobias/data/gta2tv/processed/scenario_1/annotations/c1.xml"
    # extract frames to inputs/images
    extract_frames(video_path, "inputs/images")
    # extract annotations to inputs/labels
    save_per_frame_annotations(annotation_path, 'inputs/labels', img_width=1920, img_height=1080)
    # split between train/val
    temporal_split_folder('inputs/images', 'inputs/labels')
