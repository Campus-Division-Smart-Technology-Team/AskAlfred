INVALID_BUILDING_NAMES = frozenset({
    "maintenance",
    "request", "requests",
    "job", "jobs",
    "ticket", "tickets",
})


def is_valid_building_name(name: str | None) -> bool:
    if not name:
        return False
    return name.strip().lower() not in INVALID_BUILDING_NAMES
