import numpy as np
from PIL import Image


def convert_tiff_to_png(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            # Convert Bayer pattern to RGB
            raw = np.array(img)
            height, width = raw.shape
            rgb = np.zeros((height // 2, width // 2, 3), dtype=np.uint8)

            # Using GRBG Bayer pattern
            rgb[:, :, 0] = raw[0::2, 1::2] / 4095 * 255  # Red
            rgb[:, :, 1] = (raw[0::2, 0::2] + raw[1::2, 1::2]) / 2 / 4095 * 255  # Green
            rgb[:, :, 2] = raw[1::2, 0::2] / 4095 * 255  # Blue

            img = Image.fromarray(rgb)
            img.save(output_path, "PNG")
    except FileNotFoundError:
        print(f"Failed to load: {input_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    input_path = "DE_BBBR667_2015-04-18_18-22-44-029089_k0.tiff"
    output_path = "driveu.png"
    convert_tiff_to_png(input_path, output_path)
    print(f"Converted {input_path} to {output_path}")
