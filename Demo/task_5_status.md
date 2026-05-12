"""
Project: HiveOS Core
File: test_audit.py
Purpose: Deterministic tests for Task 5 audit payload generation.
"""

import pytest

from src.audit import build_audit_payload, build_field_trace


def test_build_field_trace_tracks_mapped_and_unmapped_fields() -> None:
    original_columns = ["Voltage (V)", "amps_now", "random_sensor"]
    cleaned_columns = ["voltage_v", "amps_now", "random_sensor"]
    normalized_columns = ["voltage_v", "current_a", "random_sensor"]
    mapped_fields = {
        "voltage_v": "voltage_v",
        "amps_now": "current_a",
    }
    unmapped_fields = ["random_sensor"]

    result = build_field_trace(
        original_columns,
        cleaned_columns,
        normalized_columns,
        mapped_fields,
        unmapped_fields,
    )

    assert result == [
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


def test_build_audit_payload_includes_summary() -> None:
    audit = build_audit_payload(
        original_columns=["Voltage (V)", "random_sensor"],
        cleaned_columns=["voltage_v", "random_sensor"],
        normalized_columns=["voltage_v", "random_sensor"],
        mapped_fields={"voltage_v": "voltage_v"},
        unmapped_fields=["random_sensor"],
    )

    assert audit["summary"] == {
        "total_columns": 2,
        "mapped_count": 1,
        "unmapped_count": 1,
    }


def test_field_trace_raises_error_when_lengths_do_not_match() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        build_field_trace(
            original_columns=["Voltage (V)"],
            cleaned_columns=["voltage_v", "amps_now"],
            normalized_columns=["voltage_v"],
            mapped_fields={"voltage_v": "voltage_v"},
            unmapped_fields=[],
        )