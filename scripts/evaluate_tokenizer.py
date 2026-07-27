from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from bharat.tokenizer import BharatBPETokenizer, BharatTokenizer, load_tokenizer
from bharat.tokenizer.evaluation import TokenizerEvaluation


def _load_tokenizer(path: str) -> BharatTokenizer:
    p = Path(path)
    if not p.exists():
        msg = f"tokenizer artifact not found: {path}"
        raise FileNotFoundError(msg)
    try:
        return BharatBPETokenizer.load(p)
    except Exception:
        pass
    loaded = load_tokenizer(path)
    if loaded is None or not isinstance(loaded, BharatTokenizer):
        msg = f"unable to load tokenizer from {path}"
        raise ValueError(msg)
    return loaded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate tokenizer on evaluation dataset")
    parser.add_argument(
        "--tokenizer",
        "-t",
        action="append",
        required=True,
        dest="tokenizers",
        help="Tokenizer artifact path (repeatable)",
    )
    parser.add_argument(
        "--name",
        "-n",
        action="append",
        dest="names",
        help="Tokenizer display name (repeatable, matches --tokenizer order)",
    )
    parser.add_argument(
        "--dataset", "-d", required=True, type=Path, help="Path to evaluation JSONL dataset"
    )
    parser.add_argument("--output-report", "-o", type=Path, help="Output report path")
    parser.add_argument(
        "--detailed-records", type=Path, help="Optional path for detailed per-record JSONL output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and compute report without writing files",
    )
    parser.add_argument("--execute", action="store_true", help="Acknowledge and confirm execution")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    tokenizer_paths = args.tokenizers
    names = args.names
    if names is None:
        names = [str(Path(p).stem) for p in tokenizer_paths]
    if len(names) != len(tokenizer_paths):
        msg = "number of --name values must match --tokenizer values"
        parser.error(msg)

    if not args.dry_run and not args.execute:
        if args.output_report:
            parser.error("use --execute to confirm output, or --dry-run to validate")
        parser.error("use --dry-run for validation, or --execute to confirm")

    tokenizers: dict[str, BharatTokenizer] = {}
    for name, path in zip(names, tokenizer_paths, strict=False):
        tokenizers[name] = _load_tokenizer(path)

    eval_engine = TokenizerEvaluation(tokenizers)
    dataset_path = args.dataset
    if not dataset_path.exists():
        msg = f"dataset not found: {dataset_path}"
        parser.error(msg)

    eval_engine.load_records(dataset_path)

    report = eval_engine.compute()

    summary_lines = [
        f"Evaluator version: {report.get('evaluator_version', '?')}",
        f"Dataset SHA-256: {report.get('input_dataset_sha256', '?')}",
        f"Tokenizers: {', '.join(report.get('tokenizer_names', []))}",
        f"Report SHA-256: {report.get('report_sha256', '?')}",
    ]
    for name in report.get("tokenizer_names", []):
        agg = report.get("aggregate", {}).get(name, {})
        summary_lines.append(
            f"  {name}: {agg.get('record_count', 0)} records, "
            f"{agg.get('token_count', 0)} tokens, "
            f"fertility={agg.get('micro_fertility', '?'):.4f}, "
            f"unknown={agg.get('unknown_token_count', 0)}, "
            f"round-trip={report.get('round_trip', {}).get(name, {}).get('exact_pass_rate', '?'):.2%}"
        )

    report_json = TokenizerEvaluation.serialize_report(report)

    if args.dry_run:
        print("--- dry-run: validation passed ---")
        for line in summary_lines:
            print(f"  {line}")
        sys.stdout.flush()
        return

    output_path = args.output_report
    if output_path is None:
        print("--- computed report (stdout) ---")
        print(report_json, end="")
        return

    tmp_path = output_path.with_name(f".{output_path.name}.{secrets.token_hex(8)}.verify.tmp")
    try:
        tmp_path.write_text(report_json, encoding="utf-8")

        loaded = json.loads(tmp_path.read_text(encoding="utf-8"))
        if loaded.get("report_sha256") != report.get("report_sha256"):
            msg = "report verification failed: hash mismatch"
            raise RuntimeError(msg)

        verified_bytes = tmp_path.read_bytes()

        try:
            with open(output_path, "xb") as f:
                f.write(verified_bytes)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            msg = f"refusing to overwrite existing file: {output_path}"
            raise FileExistsError(msg) from None

        final_bytes = output_path.read_bytes()
        if final_bytes != verified_bytes:
            output_path.unlink()
            msg = f"byte-verification failed after final write to {output_path}"
            raise RuntimeError(msg)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if args.detailed_records:
        detailed = args.detailed_records
        if detailed.exists():
            msg = f"refusing to overwrite existing file: {detailed}"
            raise FileExistsError(msg)
        records_out: list[dict] = []
        for name in tokenizers:
            eval_metrics = eval_engine._metrics.get(name, [])
            for m in eval_metrics:
                records_out.append(
                    {
                        "tokenizer": name,
                        "record_id": m.record_id,
                        "char_count": m.char_count,
                        "token_count": m.token_count,
                        "tokens_per_char": round(m.tokens_per_char, 6),
                        "unknown_token_count": m.unknown_token_count,
                        "exact_round_trip": m.exact_round_trip,
                        "nfc_round_trip": m.nfc_round_trip,
                    }
                )
        detailed.write_text(
            "".join(
                json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
                for r in records_out
            ),
            encoding="utf-8",
        )

    success = {
        "status": "ok",
        "output": str(output_path),
        "report_sha256": report.get("report_sha256"),
    }
    print(json.dumps(success, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
