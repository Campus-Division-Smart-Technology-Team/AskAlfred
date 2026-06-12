from __future__ import annotations

from typing import Any, Optional

REQUIRED_ACL_FIELDS = ("tenant_id", "access_level", "allowed_roles")


def combine_pinecone_filters(
    left: Optional[dict[str, Any]],
    right: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Combine two Pinecone filters with a logical AND."""
    filters = [filter_dict for filter_dict in (left, right) if filter_dict]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]

    clauses: list[dict[str, Any]] = []
    for filter_dict in filters:
        if (
            isinstance(filter_dict, dict)
            and set(filter_dict.keys()) == {"$and"}
            and isinstance(filter_dict["$and"], list)
        ):
            clauses.extend(filter_dict["$and"])
        else:
            clauses.append(filter_dict)
    return {"$and": clauses}


def filter_authorized_structured_matches(
    matches: list[dict[str, Any]],
    access_filter: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Fail closed on missing ACL metadata and drop matches outside access scope."""
    authorised_matches: list[dict[str, Any]] = []
    for match in matches:
        metadata = match.get("metadata", {}) or {}
        # Only enforce the ACL envelope when an access filter is in play;
        # legacy vectors without ACL metadata remain visible to unscoped
        # (anonymous/dev) sessions until they are re-ingested.
        if access_filter and not has_required_acl_metadata(metadata):
            continue
        if access_filter and not metadata_matches_filter(metadata, access_filter):
            continue
        authorised_matches.append(match)
    return authorised_matches


def filter_authorized_matches(
    matches: list[dict[str, Any]],
    access_filter: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Generic alias for fail-closed match filtering across retrieval paths."""
    return filter_authorized_structured_matches(matches, access_filter=access_filter)


def has_required_acl_metadata(metadata: dict[str, Any]) -> bool:
    """Return True only when the minimum structured ACL envelope is present."""
    for field in REQUIRED_ACL_FIELDS:
        value = metadata.get(field)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if field == "allowed_roles":
            if not isinstance(value, (list, tuple, set)):
                return False
            if not any(str(role).strip() for role in value):
                return False
    return True


def apply_acl_defaults(
    metadata: dict[str, Any],
    *,
    tenant_id: Optional[str] = None,
    access_level: Optional[str] = None,
    allowed_roles: Optional[list[str] | tuple[str, ...]] = None,
) -> dict[str, Any]:
    """Stamp default ACL fields into metadata when they are absent."""
    if tenant_id and not metadata.get("tenant_id"):
        metadata["tenant_id"] = tenant_id
    if access_level and not metadata.get("access_level"):
        metadata["access_level"] = access_level
    if allowed_roles is not None and not metadata.get("allowed_roles"):
        metadata["allowed_roles"] = [
            str(role).strip() for role in allowed_roles if str(role).strip()
        ]
    return metadata


def metadata_matches_filter(
    metadata: dict[str, Any], filter_dict: dict[str, Any]
) -> bool:
    """Evaluate a small Pinecone-style metadata filter against a record."""
    if not filter_dict:
        return True

    if "$and" in filter_dict:
        clauses = filter_dict.get("$and") or []
        return all(
            metadata_matches_filter(metadata, clause)
            for clause in clauses
            if isinstance(clause, dict)
        )

    if "$or" in filter_dict:
        clauses = filter_dict.get("$or") or []
        return any(
            metadata_matches_filter(metadata, clause)
            for clause in clauses
            if isinstance(clause, dict)
        )

    for field, condition in filter_dict.items():
        if field.startswith("$"):
            return False
        if not _value_matches_condition(metadata.get(field), condition):
            return False
    return True


def _value_matches_condition(value: Any, condition: Any) -> bool:
    if isinstance(condition, dict):
        if "$eq" in condition:
            expected = condition["$eq"]
            return value == expected
        if "$in" in condition:
            options = condition["$in"] or []
            if isinstance(value, (list, tuple, set)):
                return any(item in options for item in value)
            return value in options
        return False
    return value == condition
