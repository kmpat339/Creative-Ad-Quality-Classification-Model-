
import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_and_augment(img_path, output_path, img_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        return
    # Basic normalization
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Example augmentation: horizontal flip
    flipped = cv2.flip(img, 1)

    # Save both original and augmented
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(output_path, f"{base_name}_norm.jpg"), img[..., ::-1])
    cv2.imwrite(os.path.join(output_path, f"{base_name}_flip.jpg"), flipped[..., ::-1])

if __name__ == "__main__":
    import argparse, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, help='Source images root')
    parser.add_argument('--dst_dir', required=True, help='Destination dir')
    args = parser.parse_args()

    pathlib.Path(args.dst_dir).mkdir(parents=True, exist_ok=True)
    for cls in os.listdir(args.src_dir):
        src_cls = os.path.join(args.src_dir, cls)
        dst_cls = os.path.join(args.dst_dir, cls)
        pathlib.Path(dst_cls).mkdir(exist_ok=True)
        for img_file in tqdm(os.listdir(src_cls)):
            preprocess_and_augment(os.path.join(src_cls, img_file), dst_cls)
