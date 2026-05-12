"""
Project: HiveOS Core
File: test_matchers.py
Purpose: Deterministic tests for Task 3 canonical field matching.
"""

from src.matchers import match_columns


def test_exact_canonical_match() -> None:
    columns = ["voltage_v", "current_a", "temp_c"]

    result = match_columns(columns)

    assert result["mapped_fields"] == {
        "voltage_v": "voltage_v",
        "current_a": "current_a",
        "temp_c": "temp_c",
    }
    assert result["unmapped_fields"] == []


def test_known_aliases_match_canonical_fields() -> None:
    columns = ["pack_voltage", "amps_now", "temperature_c"]

    result = match_columns(columns)

    assert result["mapped_fields"] == {
        "pack_voltage": "voltage_v",
        "amps_now": "current_a",
        "temperature_c": "temp_c",
    }
    assert result["unmapped_fields"] == []


def test_unknown_fields_remain_unmapped() -> None:
    columns = ["voltage_v", "random_sensor"]

    result = match_columns(columns)

    assert result["mapped_fields"] == {
        "voltage_v": "voltage_v",
    }
    assert result["unmapped_fields"] == ["random_sensor"]


def test_empty_column_list_returns_empty_results() -> None:
    result = match_columns([])

    assert result["mapped_fields"] == {}
    assert result["unmapped_fields"] == []


def test_no_guessing_for_ambiguous_fields() -> None:
    columns = ["battery_data", "sensor_value"]

    result = match_columns(columns)

    assert result["mapped_fields"] == {}
    assert result["unmapped_fields"] == ["battery_data", "sensor_value"]


def test_mapping_is_deterministic() -> None:
    columns = ["pack_voltage", "amps_now", "random_sensor"]

    first_result = match_columns(columns)
    second_result = match_columns(columns)

    assert first_result == second_result


def test_duplicate_canonical_targets_do_not_both_map() -> None:
    columns = ["voltage_v", "pack_voltage"]

    result = match_columns(columns)

    assert result["mapped_fields"] == {
        "voltage_v": "voltage_v",
    }
    assert result["unmapped_fields"] == ["pack_voltage"]