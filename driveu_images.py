import json
import os
import random
import shutil


def make_simlinks(file_path, split):
    with open(file_path, "r") as file:
        data = json.load(file)
        if "images" in data and isinstance(data["images"], list):
            for image in data["images"]:
                if "image_path" in image:
                    old_path = image["image_path"]
                    new_path = os.path.join("/home/guest/Datasets/DriveU", old_path)
                    new_path = os.path.normpath(new_path)

                    # Create symlink
                    symlink_path = os.path.join(
                        f"./dataset/images/{split}", os.path.basename(old_path)
                    )
                    symlink_path = os.path.normpath(symlink_path)
                    if not os.path.exists(symlink_path):
                        os.symlink(new_path, symlink_path)

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
    make_simlinks(file_path, split)

    split = "test"
    file_path = f"/home/guest/Datasets/DriveU/v2.0/DTLD_{split}.json"
    make_simlinks(file_path, split)

    move_validation_images("train", 5.0)
