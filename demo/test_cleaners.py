HiveOS Core

HiveOS Core is a deterministic battery data normalization engine that converts raw battery datasets into a clean, standardized, and auditable format.

⸻

Current Scope

This project is currently implementing:

* Task 1: File Ingestion and DataFrame Loading (MVP) ✅ Complete
* Task 2: Column Name Cleaning Layer (MVP) ✅ Complete
* Task 3: Canonical Schema & Matching Layer (MVP) ✅ Complete
* Task 4: DataFrame Normalization Layer (MVP) ✅ Complete
* Task 5: Audit & Traceability Layer (MVP) ✅ Complete

⸻

Supported Input

* .csv
* .json

⸻

Current Pipeline

File Upload
    ↓
Loader (load_dataframe)
    ↓
Cleaner (clean_colname)
    ↓
Matcher (match_columns)
    ↓
Normalizer (normalize_dataframe)
    ↓
Audit (build_audit_payload)
    ↓
Structured API Response

⸻

Current Public Interface

Backend

from src.loaders import load_dataframe
from src.cleaners import clean_colname
from src.matchers import match_columns
from src.normalizers import normalize_dataframe
from src.audit import build_audit_payload
from src.core import process_file

API

POST /api/load-file

Returns a structured response including:

* file metadata
* column transformations
* mapping results
* audit trace
* warnings
* data preview

⸻

Output Structure (MVP)

Example response:

{
  "status": "success",
  "schema_version": "core-schema-v0.1",
  "mapping_version": "core-mapping-v0.1",
  "file": {
    "name": "data.csv",
    "row_count": 100,
    "column_count": 3
  },
  "columns": {
    "original": ["Voltage (V)", "amps_now", "random_sensor"],
    "cleaned": ["voltage_v", "amps_now", "random_sensor"],
    "normalized": ["voltage_v", "current_a", "random_sensor"]
  },
  "mapping": {
    "mapped_fields": {
      "voltage_v": "voltage_v",
      "amps_now": "current_a"
    },
    "unmapped_fields": ["random_sensor"]
  },
  "audit": {
    "field_trace": [
      {
        "original": "Voltage (V)",
        "cleaned": "voltage_v",
        "normalized": "voltage_v",
        "status": "mapped"
      }
    ],
    "summary": {
      "total_columns": 3,
      "mapped_count": 2,
      "unmapped_count": 1
    }
  },
  "warnings": ["Some fields could not be mapped."],
  "data_preview": [
    {
      "voltage_v": 3.7,
      "current_a": 1.2,
      "random_sensor": 999
    }
  ],
  "validation": "File processed, normalized, and audited successfully."
}

⸻

Design Principles

HiveOS Core is built with strict constraints:

* deterministic behavior (same input → same output)
* no hidden transformations
* no guessing or probabilistic mapping
* full traceability of all changes
* strict separation of concerns between layers

⸻

Layer Responsibilities

Layer	Responsibility
Loader	Load file into DataFrame
Cleaner	Standardize column names
Matcher	Map cleaned names to canonical schema
Normalizer	Apply canonical names to DataFrame
Audit	Record and explain all transformations
Core	Orchestrate full pipeline

⸻

What HiveOS Core Does NOT Do (MVP)

* does not calculate SOC / SOH
* does not perform unit conversion
* does not modify data values
* does not drop or infer columns
* does not use machine learning
* does not store data

⸻

System Status

Backend:        Functional MVP ✅
Frontend:       Functional MVP ✅
Pipeline:       Fully connected ✅
Audit Layer:    Implemented ✅
Testing:        Layer-level coverage ✅

⸻

Next Phase

Planned improvements:

* enhanced audit metadata (timestamps, IDs)
* expanded canonical schema
* alias expansion based on real-world datasets
* UI improvements (mapping + audit visualization)
* optional data validation layer
* API versioning and stability guarantees