"""Single-source definitions for image-space annotation support."""
from __future__ import annotations


def normalized_max_side(x1: float, y1: float, x2: float, y2: float, image_width: float, image_height: float) -> float:
    """Return the anisotropically normalized max-side support.

    The horizontal and vertical box supports are divided by their respective
    image dimensions before taking the maximum: ``max(w/W, h/H)``.  This is
    deliberately not ``max(w,h)/max(W,H)``, which can understate a tall box in
    a wide image (or a wide box in a tall image).
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    return max(w / float(image_width), h / float(image_height))


def normalized_max_side_from_box(box, image_width: float, image_height: float) -> float:
    """Variant for inventory box records exposing x/y coordinates."""
    return normalized_max_side(box.x1, box.y1, box.x2, box.y2, image_width, image_height)
