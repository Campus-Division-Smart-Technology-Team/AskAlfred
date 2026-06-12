import logging

from building import normalise_building_name
from building.filename_building_parser import FilenameBuildingResolver
from ingest.document_content import (
    extract_maintenance_csv,
    is_fire_risk_assessment,
    load_building_names_with_aliases,
    split_alias_field,
)


class _Cache:
    def __init__(self):
        self.name_mapping = {}
        self.alias_mapping = {}

    def get_name_mapping(self):
        return self.name_mapping

    def get_alias_mapping(self):
        return self.alias_mapping

    def update_from_csv(self, name_mapping, alias_mapping, _metadata_cache):
        self.name_mapping = name_mapping
        self.alias_mapping = alias_mapping


class _Context:
    def __init__(self):
        self.cache = _Cache()
        self.logger = logging.getLogger(__name__)


def test_fire_risk_assessment_detection_requires_fire_for_risk_assessment():
    assert is_fire_risk_assessment("coshh-risk-assessment.pdf") is False
    assert is_fire_risk_assessment("FM-FRA-SenateHouse-2025-03.pdf") is True
    assert is_fire_risk_assessment("fire-risk-assessment.pdf") is True


def test_maintenance_csv_skips_blank_building_rows():
    data = (
        "Buildings,Hard FM Services Jobs - Complete\n"
        "Senate House,1\n"
        "   ,2\n"
    ).encode("utf-8")

    docs = extract_maintenance_csv("Maintenance Jobs.csv", data, {})

    assert len(docs) == 1
    assert docs[0][1] == "Senate House"


def test_property_alias_loader_preserves_first_exact_alias(tmp_path):
    csv_path = tmp_path / "Dim-Property.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Property name,Property names,Property alternative names,UsrFRACondensedPropertyName",
                '65 Avon Street,65 Avon Street,"Retort House, The Sheds, BDFI",',
                '65 Avon Street Data Centre,65 Avon Street and Data Centre,"Retort House, The Sheds, BDFI",',
            ]
        ),
        encoding="utf-8",
    )

    _, _, aliases = load_building_names_with_aliases(
        _Context(),
        str(tmp_path),
        "Dim-Property.csv",
    )

    assert aliases["retort"] == "65 Avon Street"


def test_split_alias_field_expands_compressed_numbered_street_aliases():
    aliases = split_alias_field(
        "36 Tyndall Park Road, 3, 5, 7, 9, 11, 13, 15, 17, "
        "19, 21 Woodland Road, Multi Media Centre"
    )

    assert "3 Woodland Road" in aliases
    assert "19 Woodland Road" in aliases
    assert "21 Woodland Road" in aliases
    assert "Multi Media Centre" in aliases


def test_street_abbreviations_normalise_only_at_the_end():
    assert normalise_building_name("21 Woodland Rd") == "21 woodland road"
    assert normalise_building_name("64-66 Avon St") == "64-66 avon street"
    assert normalise_building_name("71 St Michaels Hill") == "71 st michaels hill"


def test_compressed_woodland_aliases_resolve_filename_variants(tmp_path):
    csv_path = tmp_path / "Dim-Property.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Property name,Property names,Property alternative names,UsrFRACondensedPropertyName",
                '"Arts Complex",,"36 Tyndall Park Road, 3, 5, 7, 9, 11, '
                '13, 15, 17, 19, 21 Woodland Road, Multi Media Centre",',
            ]
        ),
        encoding="utf-8",
    )

    canonicals, names, aliases = load_building_names_with_aliases(
        _Context(),
        str(tmp_path),
        "Dim-Property.csv",
    )
    resolver = FilenameBuildingResolver(names, aliases, canonicals)

    assert aliases["19 woodland road"] == "Arts Complex"
    assert (
        resolver.resolve("FM-FRA-19WoodlandRoad-2024-3.pdf", "").canonical
        == "Arts Complex"
    )
    assert (
        resolver.resolve("FM-FRA-21WoodlandRd-2023-3.pdf", "").canonical
        == "Arts Complex"
    )


def test_exact_alias_beats_suffix_stripped_generic_key(tmp_path):
    csv_path = tmp_path / "Dim-Property.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Property name,Property names,Property alternative names,UsrFRACondensedPropertyName",
                "Goldney Tower,Goldney Tower,,",
                "Goldney Hall A - J,Goldney Hall,,",
            ]
        ),
        encoding="utf-8",
    )

    canonicals, names, aliases = load_building_names_with_aliases(
        _Context(),
        str(tmp_path),
        "Dim-Property.csv",
    )
    resolver = FilenameBuildingResolver(names, aliases, canonicals)

    result = resolver.resolve("RFM-FRA-GoldneyHall-2026-01.pdf", "")

    assert aliases["goldney hall"] == "Goldney Hall A - J"
    assert result.canonical == "Goldney Hall A - J"
