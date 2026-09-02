# Source data

Place the supplied `delivery_data.csv` in this directory. Raw CSV files are intentionally ignored because the supplied extract is 54 MB and may have separate distribution terms.

Expected command:

```bash
python -m src.pipeline --input data/raw/delivery_data.csv --run-all
```

The exact required fields and aggregation rules are documented in `docs/DATA_DICTIONARY.md`. `--generate-demo` produces a deterministic contract-compatible sample for CI; it does not replace the real analysis.
