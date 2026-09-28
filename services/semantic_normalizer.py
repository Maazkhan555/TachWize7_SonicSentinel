"""
services/semantic_normalizer.py

Maps raw model labels from different label systems (FSD50K, AudioSet-527)
to a shared normalized event vocabulary.

NEVER compare raw label strings across models directly.
Always normalize first, then compare normalized tokens.

Usage:
    from services.semantic_normalizer import normalize_label, NORMALIZED_EVENTS

    norm = normalize_label("Police car (siren)")   # → "SIREN"
    norm = normalize_label("Siren")                # → "SIREN"
    norm = normalize_label("Bicycle bell")         # → "BICYCLE"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Mapping: substring (lowercase) → NORMALIZED_EVENT
# Checked in order — first match wins.
# ---------------------------------------------------------------------------
_RULES: list[tuple[str, str]] = [
    # ── Emergency / Alert ──────────────────────────────────────────────────
    ("siren",               "SIREN"),
    ("alarm",               "ALARM"),
    ("emergency",           "ALARM"),
    ("fire alarm",          "ALARM"),
    ("smoke detector",      "ALARM"),
    ("civil defense",       "ALARM"),
    ("air horn",            "ALARM"),
    ("klaxon",              "ALARM"),

    # ── Gunshot / Explosion ────────────────────────────────────────────────
    ("gunshot",             "GUNSHOT"),
    ("gunfire",             "GUNSHOT"),
    ("gun",                 "GUNSHOT"),
    ("explosion",           "EXPLOSION"),
    ("blast",               "EXPLOSION"),
    ("boom",                "EXPLOSION"),
    ("firework",            "EXPLOSION"),
    ("firecracker",         "EXPLOSION"),

    # ── Glass / Impact ─────────────────────────────────────────────────────
    ("glass",               "GLASS_BREAK"),
    ("shatter",             "GLASS_BREAK"),
    ("breaking",            "GLASS_BREAK"),
    ("crash",               "IMPACT"),
    ("thump",               "IMPACT"),
    ("thud",                "IMPACT"),
    ("slam",                "IMPACT"),
    ("bang",                "IMPACT"),

    # ── Scream / Distress ──────────────────────────────────────────────────
    ("scream",              "SCREAM"),
    ("shriek",              "SCREAM"),
    ("shout",               "SCREAM"),
    ("yell",                "SCREAM"),
    ("crying",              "CRYING"),
    ("sobbing",             "CRYING"),
    ("cry",                 "CRYING"),
    ("whimper",             "CRYING"),

    # ── Vehicle ────────────────────────────────────────────────────────────
    ("police car",          "VEHICLE"),
    ("ambulance",           "VEHICLE"),
    ("fire engine",         "VEHICLE"),
    ("fire truck",          "VEHICLE"),
    ("motorcycle",          "VEHICLE"),
    ("motor vehicle",       "VEHICLE"),
    ("car",                 "VEHICLE"),
    ("truck",               "VEHICLE"),
    ("bus",                 "VEHICLE"),
    ("vehicle",             "VEHICLE"),
    ("traffic",             "VEHICLE"),
    ("engine",              "VEHICLE"),
    ("train",               "VEHICLE"),
    ("aircraft",            "VEHICLE"),
    ("airplane",            "VEHICLE"),
    ("helicopter",          "VEHICLE"),
    ("boat",                "VEHICLE"),

    # ── Horn / Bell ────────────────────────────────────────────────────────
    ("horn",                "HORN"),
    ("honk",                "HORN"),
    ("beep",                "HORN"),
    ("bicycle bell",        "BICYCLE"),
    ("bicycle",             "BICYCLE"),
    ("bell",                "BELL"),
    ("chime",               "BELL"),
    ("doorbell",            "BELL"),
    ("church bell",         "BELL"),
    ("cowbell",             "BELL"),

    # ── Animal ─────────────────────────────────────────────────────────────
    ("dog",                 "ANIMAL"),
    ("bark",                "ANIMAL"),
    ("cat",                 "ANIMAL"),
    ("meow",                "ANIMAL"),
    ("bird",                "ANIMAL"),
    ("animal",              "ANIMAL"),
    ("insect",              "ANIMAL"),
    ("cricket",             "ANIMAL"),
    ("frog",                "ANIMAL"),
    ("livestock",           "ANIMAL"),
    ("fowl",                "ANIMAL"),
    ("crow",                "ANIMAL"),
    ("growl",               "ANIMAL"),
    ("purr",                "ANIMAL"),

    # ── Speech / Voice ─────────────────────────────────────────────────────
    ("speech",              "SPEECH"),
    ("speaking",            "SPEECH"),
    ("conversation",        "SPEECH"),
    ("chatter",             "SPEECH"),
    ("whispering",          "SPEECH"),
    ("singing",             "SPEECH"),
    ("laughter",            "SPEECH"),
    ("giggle",              "SPEECH"),
    ("applause",            "SPEECH"),
    ("crowd",               "SPEECH"),
    ("human voice",         "SPEECH"),
    ("male speech",         "SPEECH"),
    ("female speech",       "SPEECH"),
    ("child speech",        "SPEECH"),

    # ── Machinery / Tools ──────────────────────────────────────────────────
    ("drill",               "MACHINERY"),
    ("saw",                 "MACHINERY"),
    ("hammer",              "MACHINERY"),
    ("power tool",          "MACHINERY"),
    ("machinery",           "MACHINERY"),
    ("machine",             "MACHINERY"),
    ("printer",             "MACHINERY"),
    ("typewriter",          "MACHINERY"),
    ("typing",              "MACHINERY"),
    ("mechanisms",          "MACHINERY"),

    # ── Music ──────────────────────────────────────────────────────────────
    ("music",               "MUSIC"),
    ("guitar",              "MUSIC"),
    ("piano",               "MUSIC"),
    ("drum",                "MUSIC"),
    ("bass",                "MUSIC"),
    ("violin",              "MUSIC"),
    ("instrument",          "MUSIC"),
    ("organ",               "MUSIC"),
    ("harmonica",           "MUSIC"),
    ("harp",                "MUSIC"),

    # ── Water / Nature ─────────────────────────────────────────────────────
    ("rain",                "NATURE"),
    ("thunder",             "NATURE"),
    ("wind",                "NATURE"),
    ("ocean",               "NATURE"),
    ("water",               "NATURE"),
    ("stream",              "NATURE"),
    ("waves",               "NATURE"),
    ("fire",                "NATURE"),

    # ── Domestic / Background ──────────────────────────────────────────────
    ("background",          "BACKGROUND"),
    ("silence",             "BACKGROUND"),
    ("noise",               "BACKGROUND"),
    ("static",              "BACKGROUND"),
    ("domestic",            "BACKGROUND"),
    ("home",                "BACKGROUND"),
    ("inside",              "BACKGROUND"),
    ("outside",             "BACKGROUND"),
]

# All known normalized event tokens
NORMALIZED_EVENTS: set[str] = {v for _, v in _RULES}


def normalize_label(raw_label: str) -> str:
    """
    Map a raw model label to a normalized event token.

    Parameters
    ----------
    raw_label : str
        The raw label string from any model (e.g. "Police car (siren)", "Siren").

    Returns
    -------
    str
        Normalized event token (e.g. "SIREN") or "UNKNOWN" if no rule matches.
    """
    if not raw_label or raw_label in ("—", ""):
        return "UNKNOWN"
    lower = raw_label.lower()
    for substring, event in _RULES:
        if substring in lower:
            return event
    return "UNKNOWN"


def normalize_predictions(top_predictions: list[dict]) -> list[dict]:
    """
    Add a 'normalized' key to each prediction dict.

    Parameters
    ----------
    top_predictions : list of {label, confidence}

    Returns
    -------
    list of {label, confidence, normalized}
    """
    return [
        {**p, "normalized": normalize_label(p.get("label", ""))}
        for p in top_predictions
    ]


def labels_agree(label_a: str, label_b: str) -> bool:
    """
    Return True if two raw labels normalize to the same event token.
    Never compares raw strings directly.
    """
    na = normalize_label(label_a)
    nb = normalize_label(label_b)
    if na == "UNKNOWN" or nb == "UNKNOWN":
        return False
    return na == nb
