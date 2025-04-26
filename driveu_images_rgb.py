import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from tiff_to_png import convert_tiff_to_png


def process_image(image, split, data_path):
    if "image_path" in image:
        old_path = image["image_path"]
        new_path = os.path.join(data_path, old_path)
        new_path = os.path.normpath(new_path)

        # Create RGB PNG image
        output_path = os.path.join(
            f"./dataset/images/{split}", os.path.basename(old_path)
        )
        output_path = os.path.normpath(output_path)
        output_path = os.path.splitext(output_path)[0] + ".png"

        # Assuming convert_tiff_to_png is defined elsewhere
        convert_tiff_to_png(new_path, output_path)


def make_png_dataset(file_path, split):
    dataset_dir = "./dataset/images"
    os.makedirs(os.path.join(dataset_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "val"), exist_ok=True)

    with open(file_path, "r") as file:
        data = json.load(file)
        if "images" in data and isinstance(data["images"], list):
            # Prepare for parallel execution
            data_path = "/home/guest/Datasets/DriveU"
            with ThreadPoolExecutor() as executor:
                # Map each image to the process_image function for parallel processing
                list(
                    tqdm(
                        executor.map(
                            lambda image: process_image(image, split, data_path),
                            data["images"],
                        ),
                        total=len(data["images"]),
                    )
                )

            # Save the modified JSON back to the file
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)


def move_validation_images(split, percentage):
    train_dir = f"./dataset/images/{split}"
    val_dir = "./dataset/images/val"
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    images = [f for f in os.listdir(train_dir)]
    num_val_images = int(len(images) * percentage / 100.0)
    val_images = random.sample(images, num_val_images)

    for image in val_images:
        src_path = os.path.join(train_dir, image)
        dest_path = os.path.join(val_dir, image)
        shutil.move(src_path, dest_path)


if __name__ == "__main__":
    split = "train"
    file_path = f"/home/guest/Datasets/DriveU/v2.0/DTLD_{split}.json"
    make_png_dataset(file_path, split)

    split = "test"
    file_path = f"/home/guest/Datasets/DriveU/v2.0/DTLD_{split}.json"
    make_png_dataset(file_path, split)

    move_validation_images("train", 5.0)
