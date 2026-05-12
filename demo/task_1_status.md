"""
Project: HiveOS Core
File: test_converters.py
Purpose: Deterministic tests for Task 7 value conversion behavior.
"""

import pandas as pd

from src.converters import (
    apply_conversions,
    convert_current_to_amps,
    convert_temperature_to_celsius,
    convert_voltage_to_volts,
    parse_number,
)


def test_parse_number_from_string() -> None:
    assert parse_number("3.820V") == 3.82
    assert parse_number("87%") == 87.0
    assert parse_number(" 25 C ") == 25.0


def test_temperature_fahrenheit_converts_to_celsius() -> None:
    value, conversion = convert_temperature_to_celsius("77F")

    assert value == 25.0
    assert conversion == "fahrenheit_to_celsius"


def test_temperature_celsius_string_becomes_number_without_conversion_label() -> None:
    value, conversion = convert_temperature_to_celsius("25 C")

    assert value == 25.0
    assert conversion is None


def test_voltage_millivolts_converts_to_volts() -> None:
    value, conversion = convert_voltage_to_volts("3820mV")

    assert value == 3.82
    assert conversion == "millivolts_to_volts"


def test_voltage_volts_string_becomes_number_without_conversion_label() -> None:
    value, conversion = convert_voltage_to_volts("3.820V")

    assert value == 3.82
    assert conversion is None


def test_current_milliamps_converts_to_amps() -> None:
    value, conversion = convert_current_to_amps("1200mA")

    assert value == 1.2
    assert conversion == "milliamps_to_amps"


def test_apply_conversions_updates_known_fields_and_tracks_audit() -> None:
    df = pd.DataFrame({
        "voltage_v": ["3820mV"],
        "current_a": ["1200mA"],
        "temp_c": ["77F"],
        "random_sensor": ["abc"],
    })

    converted_df, conversion_audit = apply_conversions(df)

    assert converted_df["voltage_v"].iloc[0] == 3.82
    assert converted_df["current_a"].iloc[0] == 1.2
    assert converted_df["temp_c"].iloc[0] == 25.0
    assert converted_df["random_sensor"].iloc[0] == "abc"

    assert conversion_audit == [
        {
            "field": "temp_c",
            "row": 0,
            "conversion": "fahrenheit_to_celsius",
            "original_value": "77F",
            "converted_value": 25.0,
        },
        {
            "field": "voltage_v",
            "row": 0,
            "conversion": "millivolts_to_volts",
            "original_value": "3820mV",
            "converted_value": 3.82,
        },
        {
            "field": "current_a",
            "row": 0,
            "conversion": "milliamps_to_amps",
            "original_value": "1200mA",
            "converted_value": 1.2,
        },
    ]


def test_apply_conversions_does_not_mutate_original_dataframe() -> None:
    df = pd.DataFrame({
        "temp_c": ["77F"],
    })

    converted_df, _ = apply_conversions(df)

    assert df["temp_c"].iloc[0] == "77F"
    assert converted_df["temp_c"].iloc[0] == 25.0