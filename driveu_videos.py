import os

import cv2
import numpy as np


def tiff_to_video(input_root, output_root, fps=1.5, bayer_to_rgb=False):
    """
    Convert 16-bit TIFF images (with 12-bit data) to 8-bit and create MP4 videos.
    Optionally, convert Bayer GRBG images to RGB.

    Args:
        input_root (str): Root folder containing subfolders with TIFF images.
        output_root (str): Root folder where the videos will be saved.
        fps (int): Frames per second for the output video.
        bayer_to_rgb (bool): If True, apply Bayer to RGB conversion before saving.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for subdir, _, files in os.walk(input_root):
        # Filter TIFF files ending with 'k0.tiff'
        tiff_files = sorted([f for f in files if f.endswith("k0.tiff")])

        if not tiff_files:
            continue  # Skip if no matching files

        # Get the folder name (last part of subdir)
        folder_name = os.path.basename(subdir)
        video_path = os.path.join(output_root, f"{folder_name}.mp4")

        first_image_path = os.path.join(subdir, tiff_files[0])
        first_frame = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)

        if first_frame is None:
            print(f"Skipping folder {folder_name}, could not read first image.")
            continue

        # Convert 16-bit Bayer image to RGB if needed, then to 8-bit
        first_frame_processed = process_image(first_frame, bayer_to_rgb)
        height, width = first_frame_processed.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        is_color = True if bayer_to_rgb else False  # Grayscale if no Bayer conversion
        video_writer = cv2.VideoWriter(
            video_path, fourcc, fps, (width, height), isColor=is_color
        )

        for tiff_file in tiff_files:
            image_path = os.path.join(subdir, tiff_file)
            frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if frame is None:
                print(f"Warning: Skipping {image_path} (could not read)")
                continue

            # Process frame (Bayer to RGB and 12-bit to 8-bit conversion)
            frame_processed = process_image(frame, bayer_to_rgb)
            video_writer.write(frame_processed)

        video_writer.release()
        print(f"Saved video: {video_path}")


def process_image(image, bayer_to_rgb):
    """
    Process an image: Convert from Bayer to RGB if needed, then scale from 12-bit to 8-bit.

    Args:
        image (numpy.ndarray): 16-bit Bayer input image.
        bayer_to_rgb (bool): If True, apply Bayer to RGB conversion.

    Returns:
        numpy.ndarray: Processed 8-bit image.
    """
    image = np.clip(image, 0, 4095)  # Ensure values are within 12-bit range

    if bayer_to_rgb:
        # Convert Bayer GRBG to RGB before scaling to 8-bit
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GRBG2BGR)

    # Convert 12-bit to 8-bit
    image_8bit = (image / 16).astype(np.uint8)

    return image_8bit


# Example usage:
# tiff_to_video("/home/guest/Datasets/DriveU", "./dataset/videos/raw")
tiff_to_video("/home/guest/Datasets/DriveU", "./dataset/videos/rgb", bayer_to_rgb=True)
