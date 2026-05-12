"""
Project: HiveOS Core
File: test_normalizers.py
Purpose: Deterministic tests for Task 4 DataFrame normalization.
"""

import pandas as pd

from src.normalizers import normalize_dataframe


def test_normalize_renames_mapped_columns() -> None:
    df = pd.DataFrame({
        "Voltage (V)": [3.7],
        "amps_now": [1.2],
    })

    cleaned_columns = ["voltage_v", "amps_now"]
    mapped_fields = {
        "voltage_v": "voltage_v",
        "amps_now": "current_a",
    }

    result = normalize_dataframe(df, cleaned_columns, mapped_fields)

    assert list(result.columns) == ["voltage_v", "current_a"]


def test_normalize_preserves_unmapped_columns() -> None:
    df = pd.DataFrame({
        "Voltage (V)": [3.7],
        "random_sensor": [999],
    })

    cleaned_columns = ["voltage_v", "random_sensor"]
    mapped_fields = {
        "voltage_v": "voltage_v",
    }

    result = normalize_dataframe(df, cleaned_columns, mapped_fields)

    assert list(result.columns) == ["voltage_v", "random_sensor"]


def test_normalize_does_not_modify_values() -> None:
    df = pd.DataFrame({
        "Voltage (V)": [3.7],
        "amps_now": [1.2],
    })

    cleaned_columns = ["voltage_v", "amps_now"]
    mapped_fields = {
        "voltage_v": "voltage_v",
        "amps_now": "current_a",
    }

    result = normalize_dataframe(df, cleaned_columns, mapped_fields)

    assert result["voltage_v"].iloc[0] == 3.7
    assert result["current_a"].iloc[0] == 1.2


def test_normalize_returns_new_dataframe_without_mutating_original() -> None:
    df = pd.DataFrame({
        "Voltage (V)": [3.7],
    })

    cleaned_columns = ["voltage_v"]
    mapped_fields = {
        "voltage_v": "voltage_v",
    }

    result = normalize_dataframe(df, cleaned_columns, mapped_fields)

    assert list(df.columns) == ["Voltage (V)"]
    assert list(result.columns) == ["voltage_v"]


def test_normalize_empty_mapping_uses_cleaned_columns() -> None:
    df = pd.DataFrame({
        "random_sensor": [999],
    })

    cleaned_columns = ["random_sensor"]
    mapped_fields = {}

    result = normalize_dataframe(df, cleaned_columns, mapped_fields)

    assert list(result.columns) == ["random_sensor"]