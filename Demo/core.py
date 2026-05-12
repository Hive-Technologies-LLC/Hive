HiveOS Core — Task 2 Status

Implemented Files

Backend

* src/cleaners.py
* tests/test_cleaners.py

⸻

Current Public Interface

Backend

* clean_colname(name)

⸻

Scope Guard

Task 2 Includes

* raw column name cleaning
* string casting
* whitespace trimming
* lowercasing
* normalization of all non-alphanumeric characters to underscores
* underscore collapsing
* leading/trailing underscore cleanup
* preservation of unit hints in cleaned output (e.g., _v, _c, _a when present)

Task 2 Excludes

* alias matching
* schema mapping
* DataFrame renaming
* audit output
* unit conversion
* duplicate detection fixes
* semantic interpretation of fields

⸻

System State After Task 2

Frontend (UI + validation)
        ↓
API (/api/load-file)
        ↓
Loader (load_dataframe)
        ↓
Cleaner (clean_colname)  ← implemented but not yet integrated into API pipeline
        ↓
Prepared column names for future matching

⸻

Definition of Done (Task 2)

Task 2 is considered complete when:

* raw column names can be cleaned consistently across varied input formats
* all listed test cases pass
* behavior is deterministic (same input → same output)
* non-alphanumeric characters are normalized reliably
* edge cases (whitespace, symbols, non-string input) are handled safely
* no matching or interpretation logic is introduced

⸻

Next Phase

Task 3 — Canonical Schema and Alias Map

Planned additions:

* define canonical fields
* define known aliases
* implement exact-match mapping layer
* track mapped vs unmapped fields
* prepare mapping output structure for API response

⸻

🧠 Key Improvements Reflected

This update captures:

* your move to regex-based cleaning (more robust)
* explicit note that cleaner is not yet integrated into API
* clearer definition of what is and isn’t part of cleaning
* alignment with next API evolution (mapping layer)
