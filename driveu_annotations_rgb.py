import json
import os
from copy import deepcopy
from pathlib import Path


def convert_to_coco(input_file, output_dir):
    os.makedirs("./dataset/annotations", exist_ok=True)

    with open(input_file, "r") as f:
        data = json.load(f)

    coco_format_template = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "off-relevant", "supercategory": "traffic_light"},
            {"id": 1, "name": "red-relevant", "supercategory": "traffic_light"},
            {"id": 2, "name": "yellow-relevant", "supercategory": "traffic_light"},
            {"id": 3, "name": "redyellow-relevant", "supercategory": "traffic_light"},
            {"id": 4, "name": "green-relevant", "supercategory": "traffic_light"},
            {"id": 5, "name": "off-irrelevant", "supercategory": "traffic_light"},
            {"id": 6, "name": "red-irrelevant", "supercategory": "traffic_light"},
            {"id": 7, "name": "yellow-irrelevant", "supercategory": "traffic_light"},
            {"id": 8, "name": "redyellow-irrelevant", "supercategory": "traffic_light"},
            {"id": 9, "name": "green-irrelevant", "supercategory": "traffic_light"},
        ],
    }

    sets = ["train", "val", "test"]
    coco_formats = {s: deepcopy(coco_format_template) for s in sets}
    annotation_id = {s: 1 for s in sets}

    state = {"off": 0, "red": 1, "yellow": 2, "red_yellow": 3, "green": 4, "unknown": 0}
    relevance = {"relevant": 0, "not_relevant": 5}

    for image_id, image_data in enumerate(data["images"]):
        image_path = image_data["image_path"]
        image_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        set_name = None

        for s in sets:
            file_path = Path(f"./dataset/images/{s}/{image_filename}")
            if file_path.exists() or file_path.is_symlink():
                set_name = s
                break

        if set_name is None:
            continue

        coco_formats[set_name]["images"].append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": 2048 // 2,
                "height": 1024 // 2,
            }
        )

        for label in image_data["labels"]:
            if (
                label["attributes"]["direction"] != "front"
                or label["attributes"]["reflection"] == "reflected"
            ):
                continue

            category_id = (
                state[label["attributes"]["state"]]
                + relevance[label["attributes"]["relevance"]]
            )

            bbox = [label["x"] // 2, label["y"] // 2, label["w"] // 2, label["h"] // 2]
            coco_formats[set_name]["annotations"].append(
                {
                    "id": annotation_id[set_name],
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": (label["w"] // 2) * (label["h"] // 2),
                    "iscrowd": 0,
                }
            )
            annotation_id[set_name] += 1

    for s in sets:
        output_file = os.path.join(output_dir, f"instances_{s}.json")
        with open(output_file, "w") as f:
            json.dump(coco_formats[s], f, indent=4)


if __name__ == "__main__":
    # Usage example
    convert_to_coco(
        "/home/guest/Datasets/DriveU/v2.0/DTLD_all.json",
        "./dataset/annotations",
    )
