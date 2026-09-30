"""One project-wide definition of ripeness classes and source label aliases."""

CLASS_NAMES = ("unripe", "semi_ripe", "ripe")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def canonical_class(name: str) -> str:
    """Translate dataset and model labels to the public three-class contract."""
    normalized = name.lower().strip().replace("_", "-").replace(" ", "-")
    aliases = {
        "unripe": "unripe", "unriped": "unripe", "green": "unripe", "immature": "unripe",
        "semi-ripe": "semi_ripe", "semiripe": "semi_ripe", "half-ripe": "semi_ripe",
        "half-ripened": "semi_ripe", "breaking": "semi_ripe", "reddish": "semi_ripe",
        "fully-ripe": "ripe", "fully-ripened": "ripe", "ripe": "ripe",
        "riped": "ripe", "red": "ripe",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown maturity class {name!r}; map it deliberately") from error
