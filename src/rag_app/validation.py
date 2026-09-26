"""The equality-filter contract supported by both storage backends."""

import math


def validate_filter(value):
    if value is None:
        return None
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("Filter must contain one field or an $and list of fields")
    key, item = next(iter(value.items()))
    if key == "$and":
        if not isinstance(item, list) or not 2 <= len(item) <= 20:
            raise ValueError("$and requires 2 to 20 equality filters")
        for condition in item:
            if not isinstance(condition, dict) or "$and" in condition:
                raise ValueError("Nested filter operators are not supported")
            validate_filter(condition)
    elif not isinstance(key, str) or not key.strip() or key.startswith("$"):
        raise ValueError("Invalid metadata field")
    elif type(item) not in (str, int, float, bool) or (
        isinstance(item, str) and not item.strip()
    ):
        raise ValueError(
            "Metadata values must be nonempty strings, numbers, or booleans"
        )
    elif isinstance(item, float) and not math.isfinite(item):
        raise ValueError("Metadata numbers must be finite")
    return value


def parse_filter(text):
    if not text or not text.strip():
        return None
    pairs = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value or key in pairs:
            raise ValueError("Use distinct key=value pairs separated by commas")
        pairs[key] = value
    return validate_filter(
        pairs if len(pairs) == 1 else {"$and": [{k: v} for k, v in pairs.items()]}
    )
