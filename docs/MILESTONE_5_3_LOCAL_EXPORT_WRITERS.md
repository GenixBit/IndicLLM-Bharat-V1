# Milestone 5.3 — Export Writer Interfaces (Foundation)

**Status:** Foundation implemented (dry-run only, no real weight serialization)

## Objective

Add export writer interfaces, a writer registry, and a deterministic
dry-run writer around the existing export-planning foundation.

## Implemented

### Export Writer Protocol (`bharat/serving/export.py`)

- `ExportWriter` — protocol with `write(plan) -> ExportResult`
- `DryRunExportWriter` — validates suffix, returns success/failure
  without touching any model weights

### Export Result

- `ExportResult` — frozen dataclass with `output_path`, `export_format`,
  `model_name`, `success`, `message`, `to_dict()`, `to_json()`

### Writer Registry

- `register_writer(export_format, writer_cls)` — register a writer class
- `get_writer(export_format)` — instantiate a registered writer
- `run_export(plan)` — convenience function that looks up and runs the writer
- Default dry-run writers registered for both `safetensors` and `gguf`

## Not Yet Implemented

- Real safetensors weight serialization
- Real GGUF weight serialization
- Model loading and state-dict iteration
- Weight dtype conversion
- Sharded export

## API

```python
from bharat.serving import (
    ExportPlan, ExportRequest, ExportResult,
    build_export_plan, run_export, register_writer,
)

plan = build_export_plan(ExportRequest(...))
result = run_export(plan)       # dry-run by default
print(result.to_json())
```

## Offline Guarantee

All components are deterministic and offline. No model weights are
loaded or written.
