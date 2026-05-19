import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


FONT_DIRS = [
    Path("/usr/share/fonts/truetype"),
    Path("/usr/share/fonts/opentype"),
]


def discover_fonts(limit=60):
    fonts = []
    for directory in FONT_DIRS:
        if directory.exists():
            fonts.extend(directory.rglob("*.ttf"))
            fonts.extend(directory.rglob("*.otf"))
    return sorted(fonts)[:limit]


def _render_character(character, font_path, size, offset_x, offset_y, blur_radius):
    image = Image.new("L", (64, 64), color=0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), size=size)
    # Text bounding boxes include font bearings; compensate to center ink.
    bbox = draw.textbbox((0, 0), character, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (64 - width) // 2 - bbox[0] + offset_x
    y = (64 - height) // 2 - bbox[1] + offset_y
    draw.text((x, y), character, fill=255, font=font)

    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    # Match the app preprocessing target shape and margin.
    image.thumbnail((22, 22), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    paste_at = ((28 - image.width) // 2, (28 - image.height) // 2)
    canvas.paste(image, paste_at)

    if blur_radius:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))

    return np.asarray(canvas, dtype=np.float32).reshape(28, 28, 1) / 255.0


def generate_synthetic_characters(samples_per_font=12, font_limit=60):
    fonts = discover_fonts(limit=font_limit)
    if not fonts:
        return

    sizes = [28, 32, 36, 42]
    offsets = [-3, -1, 0, 2]
    blurs = [0, 0.35]

    for font_index, font_path in enumerate(fonts):
        for label, character in enumerate(string.ascii_uppercase):
            for sample_index in range(samples_per_font):
                # Deterministic variation keeps training repeatable.
                size = sizes[(font_index + sample_index) % len(sizes)]
                offset_x = offsets[(font_index * 3 + sample_index) % len(offsets)]
                offset_y = offsets[(font_index + sample_index * 2) % len(offsets)]
                blur = blurs[(font_index + sample_index) % len(blurs)]
                yield (
                    _render_character(character, font_path, size, offset_x, offset_y, blur),
                    np.int32(label),
                )
