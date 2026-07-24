# Milestone 5.3 — Local Export Format Planning

**Status:** Implemented foundation

## Objective

Provide a safe, deterministic export-planning contract for local Bharat checkpoints targeting `safetensors` and GGUF output paths.

## Implemented

- `ExportRequest` validates local checkpoint/output paths, model name, format, and file suffix.
- `ExportPlan` provides stable dictionary and JSON serialization.
- `build_export_plan()` creates a dry-run plan without reading or writing model weights.
- Remote paths such as HTTP, S3, and Google Cloud Storage are rejected.
- Focused deterministic tests cover both formats and invalid inputs.

## Safety boundary

This milestone does not convert, serialize, upload, or publish real model weights. It adds the validated local contract required before format-specific writers are introduced. No network access, downloads, external conversion tools, or artifact uploads are used.

## Example

```python
from pathlib import Path

from bharat.serving import ExportRequest, build_export_plan

plan = build_export_plan(
    ExportRequest(
        checkpoint_path=Path("checkpoints/bharat"),
        output_path=Path("exports/bharat.safetensors"),
        export_format="safetensors",
        model_name="bharat-local",
    )
)
print(plan.to_json())
```
