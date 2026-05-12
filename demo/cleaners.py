HiveOS Core — Task 3 Status

Implemented Files

Backend

* src/matchers.py
* tests/test_matchers.py

⸻

Current Public Interface

Backend

* match_columns(columns)

⸻

Scope Guard

Task 3 Includes

* canonical field definition for MVP
* known alias mapping
* exact canonical field matching
* mapped field tracking
* unmapped field tracking
* duplicate canonical target protection
* deterministic mapping behavior
* conservative matching logic

Task 3 Excludes

* raw column cleaning
* DataFrame renaming
* normalization pipeline
* unit conversion
* audit payload generation
* fuzzy matching
* automatic alias learning
* SOC / SOH calculations
* semantic guessing of ambiguous fields

⸻

System State After Task 3

Frontend (UI + validation)
        ↓
API (/api/load-file)
        ↓
Core Pipeline (process_file)
        ↓
Loader (load_dataframe)
        ↓
Cleaner (clean_colname)
        ↓
Matcher (match_columns)
        ↓
Mapped + Unmapped Field Summary
        ↓
API Response
        ↓
Frontend Display

⸻

Definition of Done (Task 3)

Task 3 is considered complete when:

* exact canonical fields map correctly
* known aliases map to approved canonical fields
* unknown fields remain unmapped
* ambiguous fields are not guessed
* duplicate canonical targets are not both mapped
* behavior is deterministic
* all matcher tests pass
* no DataFrame renaming or normalization logic is introduced

⸻

Next Phase

Task 4 — Normalization Layer

Planned additions:

* apply mapped fields to DataFrame columns
* rename matched columns to canonical schema names
* preserve unmapped fields safely
* return normalized column list
* prepare normalized dataset preview for API response
