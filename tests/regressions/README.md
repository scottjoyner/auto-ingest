# Regression fixtures

This directory is the permanent landing zone for bugs reproduced on real machines.
Every `*.json` file is discovered automatically by `tests/test_regression_harness.py`.

## Capture workflow

1. Reproduce the bug on the affected machine.
2. Generate a sanitized environment bundle:

   ```bash
   python -m auto_ingest.diagnostics --output auto-ingest-diagnostics.tar.gz
   ```

3. Reduce the failing input to the smallest JSON-compatible payload possible.
4. Add a fixture file in this directory using an allowlisted target from
   `auto_ingest.regression.ADAPTERS`.
5. Make the fixture fail before the production fix, then pass after the fix.
6. Keep the fixture permanently; do not replace it with a looser unit test.

## Fixture format

```json
{
  "version": 1,
  "cases": [
    {
      "name": "descriptive production regression",
      "target": "queue.job",
      "input": {
        "version": 1,
        "job_id": "example",
        "profile": "sync",
        "created_at": 1700000000,
        "metadata": {}
      },
      "expect": {
        "contains": {
          "profile": "sync"
        }
      }
    }
  ]
}
```

A case may use `"raises": "ExceptionClassName"` instead of `expect` when the
correct behavior is rejection.

## Safety

Regression fixtures cannot name arbitrary Python functions or shell commands.
Only adapters explicitly registered in `auto_ingest.regression.ADAPTERS` may run.
Do not put credentials, tokens, personal data, or large media files into fixtures.
Use the diagnostics bundle for environment comparison; it reports secret presence
without recording secret values.
