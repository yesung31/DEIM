import numpy as np
from PIL import Image


def convert_tiff_to_png(input_path, output_path):
    with Image.open(input_path) as img:
        # Convert Bayer pattern to RGB
        raw = np.array(img)
        height, width = raw.shape
        rgb = np.zeros((height // 2, width // 2, 3), dtype=np.uint8)

        # GRBG Bayer pattern with bilinear interpolation
        # Create full-size arrays for each color
        red = np.zeros((height, width), dtype=np.float32)
        green = np.zeros((height, width), dtype=np.float32)
        blue = np.zeros((height, width), dtype=np.float32)

        # Fill known values
        red[0::2, 1::2] = raw[0::2, 1::2]  # R locations
        green[0::2, 0::2] = raw[0::2, 0::2]  # G locations (top)
        green[1::2, 1::2] = raw[1::2, 1::2]  # G locations (bottom)
        blue[1::2, 0::2] = raw[1::2, 0::2]  # B locations

        # Interpolate red
        red[1:-2:2, 1::2] = (red[0::2, 1::2][:-1, :] + red[2::2, 1::2]) / 2  # vertical
        red[::, 0:-2:2] = (
            red[::, 1::2][:, :-1] + red[::, 1::2][:, 1:]
        ) / 2  # horizontal

        # Interpolate blue
        blue[0:-2:2, 0::2] = (
            blue[1::2, 0::2][:-1, :] + blue[1::2, 0::2][1:, :]
        ) / 2  # vertical
        blue[::, 1:-2:2] = (
            blue[::, 0::2][:, :-1] + blue[::, 0::2][:, 1:]
        ) / 2  # horizontal

        # Interpolate green
        green[1:-2:2, 0::2] = (
            green[0::2, 0::2][:-1, :] + green[2::2, 0::2]
        ) / 2  # vertical
        green[0::2, 1:-2:2] = (
            green[0::2, 0::2][:, :-1] + green[0::2, 0::2][:, 1:]
        ) / 2  # horizontal

        # Combine channels and normalize
        rgb = np.stack((red, green, blue), axis=-1) / 4095 * 255
        rgb = rgb.astype(np.uint8)

        img = Image.fromarray(rgb)
        img.save(output_path, "PNG")


if __name__ == "__main__":
    input_path = "DE_BBBR667_2015-04-18_18-22-44-029089_k0.tiff"
    output_path = "driveu.png"
    convert_tiff_to_png(input_path, output_path)
    print(f"Converted {input_path} to {output_path}")
