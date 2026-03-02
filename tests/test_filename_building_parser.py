import pytest

from filename_building_parser import extract_building_from_filename
from building.utils import extract_building_from_query


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("FM-FRA-11 Priory Road - 2024-09.pdf", "11 Priory Road"),
        ("FM-OAS-8-10BerkeleySquare-09-2024.pdf", "8-10 Berkeley Square"),
        ("FM-FRA-64-66 Avon St vacant Bldg FRA 2025-03.pdf", "64-66 Avon St"),
    ],
)
def test_extract_building_from_filename_hybrid_fallback(filename, expected):
    assert extract_building_from_filename(filename) == expected


def test_extract_building_from_query_uses_provided_known_buildings_when_cache_empty(monkeypatch):
    monkeypatch.setattr(
        "building.utils.BuildingCacheManager.is_populated",
        lambda: False,
    )
    result = extract_building_from_query(
        "any update for Clifton Hill House?",
        known_buildings=["Clifton Hill House"],
        use_cache=True,
    )
    assert result == "Clifton Hill House"
