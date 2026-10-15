import os
import cv2
import torch
import re
from PIL import Image
from torchvision import transforms
import numpy as np
import imageio.v3 as iio

# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']
nr_metrics = ['clipiqa', 'brisque', 'niqe', 'musiq']
video_metrics  = ['dover']

def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def is_video_file(filename: str):
    return any(filename.lower().endswith(ext) for ext in video_exts)

def rgb_to_y(img):
    # Assumes img is [1, 3, H, W] in [0,1], returns [1, 1, H, W]
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 0.0625
    return y

def crop_border(img, crop):
    return img[:, :, crop:-crop, crop:-crop]

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(Image.fromarray(rgb)))
    cap.release()
    return torch.stack(frames)


def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)


def load_sequence(path: str):
    if os.path.isdir(path):
        return read_image_folder(path)
    elif os.path.isfile(path):
        if is_video_file(path):
            return read_video_frames(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Treat image as a single-frame video
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)  # [1, C, H, W]
    raise ValueError(f"Unsupported input: {path}")


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() 
        for text in re.split(r'(\d+)', s)
    ]

def img2video(subfolder_path, output_path, fps=8):
    # 2025.4.19
    img_tensor = read_image_folder(subfolder_path)
    if img_tensor is None:
        print(f"Failed to read images from {subfolder_path}")
        return
    img_tensor = img_tensor.permute(0, 2, 3, 1)  # [F, H, W, C]
    frames = (img_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()  # [F, H, W, C]
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264rgb',
        pixelformat='rgb24',
        macro_block_size=None,
        ffmpeg_params=['-crf', '0'],
    )
    print(f"Video saved to {output_path}")
