"""
Project: HiveOS Core
File: test_core_pipeline.py
Purpose: End-to-end tests for the HiveOS Core processing pipeline.
"""

import json
from pathlib import Path

from src.core import process_file


def test_core_pipeline_processes_csv_end_to_end(tmp_path: Path) -> None:
    csv_path = tmp_path / "battery_data.csv"
    csv_path.write_text(
        "Voltage (V),amps_now,random_sensor\n3.7,1.2,999\n",
        encoding="utf-8",
    )

    result = process_file(csv_path)

    assert result["status"] == "success"

    assert result["file"]["name"] == "battery_data.csv"
    assert result["file"]["row_count"] == 1
    assert result["file"]["column_count"] == 3

    assert result["columns"]["original"] == [
        "Voltage (V)",
        "amps_now",
        "random_sensor",
    ]

    assert result["columns"]["cleaned"] == [
        "voltage_v",
        "amps_now",
        "random_sensor",
    ]

    assert result["columns"]["normalized"] == [
        "voltage_v",
        "current_a",
        "random_sensor",
    ]

    assert result["mapping"]["mapped_fields"] == {
        "voltage_v": "voltage_v",
        "amps_now": "current_a",
    }

    assert result["mapping"]["unmapped_fields"] == ["random_sensor"]

    assert result["audit"]["summary"] == {
        "total_columns": 3,
        "mapped_count": 2,
        "unmapped_count": 1,
        "conversion_count": 0,
    }

    assert result["audit"]["conversions"] == []

    assert result["warnings"] == ["Some fields could not be mapped."]

    assert result["data_preview"] == [
        {
            "voltage_v": 3.7,
            "current_a": 1.2,
            "random_sensor": 999,
        }
    ]


def test_core_pipeline_processes_json_end_to_end(tmp_path: Path) -> None:
    json_path = tmp_path / "battery_data.json"
    payload = [
        {
            "Voltage (V)": 3.7,
            "amps_now": 1.2,
            "Temp[C]": 25,
        }
    ]

    json_path.write_text(json.dumps(payload), encoding="utf-8")

    result = process_file(json_path)

    assert result["status"] == "success"

    assert result["columns"]["cleaned"] == [
        "voltage_v",
        "amps_now",
        "temp_c",
    ]

    assert result["columns"]["normalized"] == [
        "voltage_v",
        "current_a",
        "temp_c",
    ]

    assert result["mapping"]["mapped_fields"] == {
        "voltage_v": "voltage_v",
        "amps_now": "current_a",
        "temp_c": "temp_c",
    }

    assert result["mapping"]["unmapped_fields"] == []
    assert result["warnings"] == []

    assert result["audit"]["summary"] == {
        "total_columns": 3,
        "mapped_count": 3,
        "unmapped_count": 0,
        "conversion_count": 0,
    }

    assert result["audit"]["conversions"] == []


def test_core_pipeline_audit_field_trace_matches_transformations(tmp_path: Path) -> None:
    csv_path = tmp_path / "battery_data.csv"
    csv_path.write_text(
        "Voltage (V),amps_now,random_sensor\n3.7,1.2,999\n",
        encoding="utf-8",
    )

    result = process_file(csv_path)

    assert result["audit"]["field_trace"] == [
        {
            "original": "Voltage (V)",
            "cleaned": "voltage_v",
            "normalized": "voltage_v",
            "status": "mapped",
        },
        {
            "original": "amps_now",
            "cleaned": "amps_now",
            "normalized": "current_a",
            "status": "mapped",
        },
        {
            "original": "random_sensor",
            "cleaned": "random_sensor",
            "normalized": "random_sensor",
            "status": "unmapped",
        },
    ]


def test_core_pipeline_tracks_value_conversions(tmp_path: Path) -> None:
    csv_path = tmp_path / "battery_data.csv"
    csv_path.write_text(
        "Voltage (V),amps_now,Temp[C]\n3820mV,1200mA,77F\n",
        encoding="utf-8",
    )

    result = process_file(csv_path)

    assert result["data_preview"] == [
        {
            "voltage_v": 3.82,
            "current_a": 1.2,
            "temp_c": 25.0,
        }
    ]

    assert result["audit"]["summary"]["conversion_count"] == 3

    assert result["audit"]["conversions"] == [
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


def test_core_pipeline_response_contains_version_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "battery_data.csv"
    csv_path.write_text(
        "Voltage (V),amps_now\n3.7,1.2\n",
        encoding="utf-8",
    )

    result = process_file(csv_path)

    assert result["schema_version"] == "core-schema-v0.1"
    assert result["mapping_version"] == "core-mapping-v0.1"