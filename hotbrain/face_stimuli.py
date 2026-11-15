"""Pre-render controlled face morphs for the short EEG feasibility pilot.

These are deliberately schematic faces. Their morph percentage describes the
rendered geometry, not the emotional state of a real person. Keeping every
non-expression pixel fixed makes the first acquisition check interpretable.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

IDENTITIES = (
    ((223, 183, 153), (51, 39, 38), 0),
    ((173, 123, 91), (33, 29, 29), 1),
    ((239, 199, 169), (102, 66, 44), 2),
    ((134, 92, 70), (36, 31, 34), 3),
)
LEVELS = (0, 20, 40, 55, 70, 80, 85, 100)


def render_face(identity: int, emotion: str, level: int, size: int = 512) -> Image.Image:
    if emotion not in ("neutral", "happy", "sad") or not 0 <= level <= 100:
        raise ValueError("Invalid face parameters")
    skin, hair, variant = IDENTITIES[identity]
    strength = 0 if emotion == "neutral" else level / 100
    im = Image.new("RGB", (size, size), (210, 211, 212))
    d = ImageDraw.Draw(im)
    dx = (-8, 6, -3, 8)[variant]
    face = (93 + dx, 58, 419 + dx, 466)
    d.ellipse((85 + dx, 260, 125 + dx, 326), fill=skin)
    d.ellipse((387 + dx, 260, 427 + dx, 326), fill=skin)
    d.ellipse(face, fill=skin, outline=(108, 83, 71), width=3)
    d.pieslice((83 + dx, 32, 427 + dx, 300), 180, 360, fill=hair)
    d.rectangle((89 + dx, 151, 109 + dx, 206), fill=hair)
    d.rectangle((403 + dx, 151, 423 + dx, 206), fill=hair)
    for x in (190 + dx, 320 + dx):
        d.ellipse((x - 27, 228, x + 27, 252), fill=(248, 247, 244), outline=(58, 49, 46), width=3)
        d.ellipse((x - 8, 232, x + 8, 249), fill=(52, 63, 67))
        d.ellipse((x - 3, 234, x + 2, 240), fill=(245, 245, 245))
    brow = -12 * strength if emotion == "sad" else 4 * strength if emotion == "happy" else 0
    d.line([(154 + dx, 204 + brow), (188 + dx, 196 - brow), (224 + dx, 205 + brow)], fill=hair, width=9, joint="curve")
    d.line([(286 + dx, 205 + brow), (320 + dx, 196 - brow), (356 + dx, 204 + brow)], fill=hair, width=9, joint="curve")
    d.line([(256 + dx, 247), (245 + dx, 319), (268 + dx, 323)], fill=(146, 105, 89), width=4)
    if emotion == "happy":
        corners = 352 - 31 * strength
        middle = 354 + 38 * strength
    elif emotion == "sad":
        corners = 352 + 26 * strength
        middle = 354 - 22 * strength
    else:
        corners = middle = 354
    points = []
    for i in range(33):
        t = i / 32
        x = 190 + dx + 130 * t
        y = middle + (corners - middle) * (2 * t - 1) ** 2
        points.append((round(x), round(y)))
    d.line(points, fill=(111, 48, 53), width=8, joint="curve")
    return im


def prepare_faces(directory: Path) -> dict[tuple[int, str, int], Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for identity in range(len(IDENTITIES)):
        for emotion in ("neutral", "happy", "sad"):
            for level in ((0,) if emotion == "neutral" else LEVELS[1:]):
                path = directory / f"face_{identity}_{emotion}_{level}.png"
                if not path.exists():
                    render_face(identity, emotion, level).save(path)
                paths[(identity, emotion, level)] = path
    return paths
