"""Canonical YOLO classes for the 106 numbered tiles and two jokers.

Class order matches the bundled ``best.pt`` checkpoint, which was trained on
raw Roboflow label text (e.g. ``Black1``, ``Black10``, ``Joker``, ``Orange1``)
sorted alphabetically rather than by tile number. Label files must use the
same order as the model's class indices, so ``CLASS_NAMES`` is sorted by that
raw form even though the entries themselves keep the readable ``black_1``
style. Do not switch this back to numeric tile order without re-running
``train.py`` (or remapping ``data/labels``) to match.
"""

COLORS = ("black", "blue", "orange", "red")


def _raw_label(canonical_name: str) -> str:
    """Roboflow-style raw label text used to derive the training class order."""
    if canonical_name == "joker":
        return "Joker"
    color, number = canonical_name.rsplit("_", 1)
    return f"{color.capitalize()}{number}"


_NUMERIC_ORDER = [f"{color}_{number}" for color in COLORS for number in range(1, 14)] + [
    "joker"
]
CLASS_NAMES = sorted(_NUMERIC_ORDER, key=_raw_label)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def canonicalize_external_name(name: str) -> str:
    """Map labels such as ``Black10`` or ``black-10`` to ``black_10``."""
    compact = "".join(character for character in name.lower() if character.isalnum())
    if compact == "joker":
        return "joker"
    for color in COLORS:
        suffix = compact[len(color) :] if compact.startswith(color) else ""
        if suffix.isdigit():
            candidate = f"{color}_{int(suffix)}"
            if candidate in CLASS_TO_ID:
                return candidate
    raise ValueError(f"Unknown external Rummikub class: {name}")


def parse_class_name(name: str) -> tuple[str | None, int | None, bool]:
    if name == "joker":
        return None, None, True
    color, number = name.rsplit("_", 1)
    if color not in COLORS or not number.isdigit() or not 1 <= int(number) <= 13:
        raise ValueError(f"Unknown Rummikub class: {name}")
    return color, int(number), False
