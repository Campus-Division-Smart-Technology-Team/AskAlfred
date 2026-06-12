import pytest

from building.alias_override import get_alias_override
from building.filename_building_parser import (
    FilenameBuildingResolver,
    extract_building_from_filename,
    load_manual_building_overrides,
    should_flag_for_review,
)
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


def test_extract_building_from_query_uses_provided_known_buildings_when_cache_empty(
    monkeypatch,
):
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


def test_alias_override_uses_normalised_keys():
    assert get_alias_override("Retort House") == "65 Avon Street"
    assert get_alias_override("retort") == "65 Avon Street"


def test_exact_alias_wins_before_fuzzy():
    resolver = FilenameBuildingResolver(
        name_to_canonical={},
        alias_to_canonical={"retort": "65 Avon Street"},
        known_buildings=["Stuart House", "65 Avon Street"],
    )

    result = resolver.resolve("UoB-Retort-House-BMS-O&M-Manual-rev1.pdf", "")

    assert result.canonical == "65 Avon Street"
    assert result.confidence == 1.0


def test_ambiguous_place_fragment_is_quarantined():
    resolver = FilenameBuildingResolver(
        name_to_canonical={},
        alias_to_canonical={},
        known_buildings=[
            "8-10 Berkeley Square",
            "12 Berkeley Square",
            "35 Berkeley Square",
        ],
    )

    result = resolver.resolve("UoB-Berkeley-Square-BMS.pdf", "")

    assert result.canonical == "Unknown"
    assert result.source == "ambiguous"
    assert should_flag_for_review(result.confidence, result.source)


def test_fuzzy_address_number_mismatch_is_quarantined():
    resolver = FilenameBuildingResolver(
        name_to_canonical={},
        alias_to_canonical={},
        known_buildings=["1 Woodland Road", "10 Woodland Road"],
    )

    result = resolver.resolve("FM-OAS-11WoodlandRoad-2025-01.pdf", "")

    assert result.canonical == "Unknown"
    assert result.source == "ambiguous"


def test_manual_overrides_are_opt_in(tmp_path):
    override_file = tmp_path / "resolved_buildings.csv"
    override_file.write_text(
        "\n".join(
            [
                "file,extracted,canonical,confidence,source",
                "generated.pdf,Retort House,Stuart House,0.75,filename",
                "manual.pdf,Retort House,65 Avon Street,1,manual",
                "ambiguous.pdf,Berkeley Square,Unknown,0,manual",
            ]
        ),
        encoding="utf-8",
    )

    overrides = load_manual_building_overrides(override_file)

    assert "generated.pdf" not in overrides
    assert overrides["manual.pdf"] == ("65 Avon Street", 1.0, "manual")
    assert overrides["ambiguous.pdf"] == ("Unknown", 0.0, "manual_unknown")
