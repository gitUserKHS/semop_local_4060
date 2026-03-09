# CP GUI Manual

## Start The GUI

```bash
.\.venv312\Scripts\python.exe cp_copilot_gui.py ^
  --episode-store data\cp_episodes.db
```

Then open:
- `http://127.0.0.1:8787`

## What The GUI Does

The CP GUI lets beginners:
- load seed contest statements
- run structured CP analysis
- inspect hidden concepts, logical frames, and DSL operators
- inspect complexity estimates
- inspect validator output and counterexamples
- inspect repair attempts
- ingest real WA/TLE/editorial incidents into the episode DB
- inspect recent episodic memory

## Recommended First Session

1. Load a seed example.
2. Run analysis.
3. Inspect `checker_kind`, `failure_type`, and `counterexample_input`.
4. Load an incident example.
5. Store it into the episode DB.
6. Re-run a related statement and inspect whether similar episodes now appear.

