import cv2
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def extract_frames(video_path, output_dir, prefix="frame"):
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
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            if box.get('outside') == '1' or box.get('occluded') == '1':
                continue
            xtl = float(box.get('xtl')) / img_width
            ytl = float(box.get('ytl')) / img_height
            xbr = float(box.get('xbr')) / img_width
            ybr = float(box.get('ybr')) / img_height

            center_x = (xtl + xbr) / 2 
            center_y = (ytl + ybr) / 2 

            if frame not in frame_annots:
                frame_annots[frame] = []
            if frame not in frame_annots_boxes:
                frame_annots_boxes[frame] = []
            frame_annots[frame].append([center_x, center_y])
            frame_annots_boxes[frame].append([xtl, ytl, xbr, ybr])

    for frame_idx, centers in frame_annots.items():
        txt_path = os.path.join(output_dir, f"{prefix}_{frame_idx:05d}.txt")
        with open(txt_path, "w") as f:
            for center_x, center_y in centers:
                f.write(f"{center_x} {center_y}\n")
    print(f"Saved annotations as .txt for {len(frame_annots)} frames to {output_dir}")

    for frame_idx, boxes in frame_annots_boxes.items():
        txt_path = os.path.join(output_dir, f"{prefix}_{frame_idx:05d}_boxes.txt")
        with open(txt_path, "w") as f:
            for xtl, ytl, xbr, ybr in boxes:
                f.write(f"{xtl} {ytl} {xbr} {ybr}\n")
    print(f"Saved box annotations as .txt for {len(frame_annots)} frames to {output_dir}")

def plot_annotations(idx, dir):
    frame_name = f"frame_{idx:05d}"
    img_path = os.path.join(dir, f"{frame_name}.jpg")
    annot_path = os.path.join(dir, f"{frame_name}.txt")

    # Load image
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Load annotations
    if os.path.exists(annot_path):
        with open(annot_path, "r") as f:
            for line in f:
                cx, cy = map(float, line.strip().split())
                print(cx, cy)
                abs_x = int(cx * width)
                abs_y = int(cy * height)
                draw.ellipse((abs_x - 5, abs_y - 5, abs_x + 5, abs_y + 5), fill="red", outline="black")
    else:
        print(f"No annotation file for frame {idx}")
    img.show()

def temporal_split_folder(inputs_dir, train_ratio=0.8, prefix="frame"):
    inputs_dir = Path(inputs_dir)
    frame_files = sorted([f for f in inputs_dir.glob("*.jpg")])

    total = len(frame_files)
    train_cutoff = int(total * train_ratio)

    train_dir = inputs_dir / "train"
    val_dir = inputs_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    for i, frame_file in enumerate(frame_files):
        target_dir = train_dir if i < train_cutoff else val_dir
        shutil.move(str(frame_file), str(target_dir / frame_file.name))

        # Move corresponding annotation .txt if it exists
        annot_file = inputs_dir / f"{frame_file.stem}.txt"
        if annot_file.exists():
            shutil.move(str(annot_file), str(target_dir / annot_file.name))

        # Move corresponding annotation .txt if it exists
        annot_file = inputs_dir / f"{frame_file.stem}_boxes.txt"
        if annot_file.exists():
            shutil.move(str(annot_file), str(target_dir / annot_file.name))

    print(f"Moved {train_cutoff} frames to 'train/', {total - train_cutoff} to 'val/'")

if __name__ == '__main__':
    video_path = "/home/tobias/data/gta2tv/processed/scenario_1/synced_camera_1.mp4"
    annotation_path = "/home/tobias/data/gta2tv/processed/scenario_1/annotations/c1.xml"
    extract_frames(video_path, "inputs")
    save_per_frame_annotations(annotation_path, 'inputs', img_width=1920, img_height=1080)
    temporal_split_folder('inputs')
    plot_annotations(100, 'inputs/train')
