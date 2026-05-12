"""
Project: HiveOS Core
File: test_loaders.py
Purpose: Deterministic tests for Task 1 file ingestion behavior.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.loaders import load_dataframe


def test_load_csv_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("voltage,current\n3.7,1.2\n", encoding="utf-8")

    df = load_dataframe(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["voltage", "current"]
    assert df.shape == (1, 2)


def test_load_json_success(tmp_path: Path) -> None:
    json_path = tmp_path / "sample.json"
    payload = [{"voltage": 3.7, "current": 1.2}]
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    df = load_dataframe(json_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["voltage", "current"]
    assert df.shape == (1, 2)


def test_missing_file_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Input file does not exist"):
        load_dataframe(missing_path)


def test_directory_path_raises_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Input path is not a file"):
        load_dataframe(tmp_path)


def test_unsupported_file_type_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataframe(file_path)


def test_uppercase_suffix_loads_successfully(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.CSV"
    csv_path.write_text("voltage,current\n3.7,1.2\n", encoding="utf-8")

    df = load_dataframe(csv_path)

    assert list(df.columns) == ["voltage", "current"]
    assert df.shape == (1, 2)


def test_invalid_json_raises_error(tmp_path: Path) -> None:
    json_path = tmp_path / "broken.json"
    json_path.write_text("{bad json}", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON format"):
        load_dataframe(json_path)


def test_empty_csv_raises_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Input file is empty"):
        load_dataframe(csv_path)


def test_header_only_csv_raises_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "headers_only.csv"
    csv_path.write_text("voltage,current\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Input file is empty"):
        load_dataframe(csv_path)