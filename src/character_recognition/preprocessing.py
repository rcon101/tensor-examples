import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image: Image.Image) -> np.ndarray:
    grayscale = ImageOps.grayscale(image)
    gray_pixels = np.asarray(grayscale, dtype=np.uint8)

    # Border pixels are a cheap estimate of the image background color.
    border = np.concatenate(
        [
            gray_pixels[0, :],
            gray_pixels[-1, :],
            gray_pixels[:, 0],
            gray_pixels[:, -1],
        ]
    )
    background_is_light = np.median(border) >= 128
    foreground = 255 - gray_pixels if background_is_light else gray_pixels
    pixels = foreground.astype(np.float32)

    # Crop to visible ink before resizing so the letter fills the 28x28 input.
    if pixels.max() > 0:
        threshold = max(30.0, pixels.max() * 0.2)
        ys, xs = np.where(pixels > threshold)
        if len(xs) and len(ys):
            left, right = xs.min(), xs.max()
            top, bottom = ys.min(), ys.max()
            foreground = foreground[top : bottom + 1, left : right + 1]

    foreground_image = Image.fromarray(foreground, mode="L")
    # Keep a small margin, matching MNIST-style centered characters.
    fitted = ImageOps.contain(foreground_image, (22, 22), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - fitted.width) // 2, (28 - fitted.height) // 2)
    canvas.paste(fitted, offset)

    normalized = np.asarray(canvas, dtype=np.float32) / 255.0
    return normalized.reshape(1, 28, 28, 1)
