"""
Project: HiveOS Core
File: test_cleaners.py
Purpose: Deterministic tests for Task 2 column name cleaning.
"""

from src.cleaners import clean_colname


def test_clean_standard_text() -> None:
    assert clean_colname("Battery Voltage") == "battery_voltage"


def test_clean_trims_whitespace() -> None:
    assert clean_colname(" Temp C ") == "temp_c"


def test_clean_replaces_punctuation() -> None:
    assert clean_colname("Voltage (V)") == "voltage_v"


def test_clean_replaces_slash_and_hyphen() -> None:
    assert clean_colname("Pack/Current-A") == "pack_current_a"


def test_clean_collapses_repeated_separators() -> None:
    assert clean_colname("cell__voltage---avg") == "cell_voltage_avg"


def test_clean_non_string_input() -> None:
    assert clean_colname(123) == "123"


def test_clean_strips_leading_and_trailing_separators() -> None:
    assert clean_colname("__Temp__") == "temp"


def test_clean_already_clean_name() -> None:
    assert clean_colname("voltage_v") == "voltage_v"


def test_clean_brackets_and_units() -> None:
    assert clean_colname("Temp[C]") == "temp_c"


def test_clean_special_characters() -> None:
    assert clean_colname("SOC%") == "soc"


def test_clean_empty_string() -> None:
    assert clean_colname("") == ""

def test_duplicate_canonical_targets_do_not_both_map() -> None:
    columns = ["voltage_v", "pack_voltage"]

    result = match_columns(columns)

    assert result["mapped_fields"] == {
        "voltage_v": "voltage_v",
    }
    assert result["unmapped_fields"] == ["pack_voltage"]