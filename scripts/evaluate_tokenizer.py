from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from bharat.tokenizer import BharatBPETokenizer, BharatTokenizer, load_tokenizer
from bharat.tokenizer.evaluation import TokenizerEvaluation


def _load_tokenizer(path: str, tokenizer_type: str) -> BharatTokenizer:
    p = Path(path)
    if not p.exists():
        msg = f"tokenizer artifact not found: {path}"
        raise FileNotFoundError(msg)

    if tokenizer_type == "bpe":
        return BharatBPETokenizer.load(p)

    if tokenizer_type == "auto":
        try:
            return BharatBPETokenizer.load(p)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
        loaded = load_tokenizer(path)
        if loaded is None or not isinstance(loaded, BharatTokenizer):
            msg = f"unable to load tokenizer from {path}"
            raise ValueError(msg)
        return loaded

    msg = f"unsupported tokenizer type: {tokenizer_type!r}"
    raise ValueError(msg)


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
        "--tokenizer-type",
        default="auto",
        choices=["bpe", "auto"],
        help="Tokenizer type (bpe, auto)",
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
    tokenizer_type = args.tokenizer_type
    names = args.names
    if names is None:
        names = [str(Path(p).stem) for p in tokenizer_paths]
    if len(names) != len(tokenizer_paths):
        msg = "number of --name values must match --tokenizer values"
        parser.error(msg)
    if len(names) != len(set(names)):
        msg = "duplicate tokenizer display names are not allowed"
        parser.error(msg)

    if not args.dry_run and not args.execute:
        if args.output_report:
            parser.error("use --execute to confirm output, or --dry-run to validate")
        parser.error("use --dry-run for validation, or --execute to confirm")

    tokenizers: dict[str, BharatTokenizer] = {}
    for name, path in zip(names, tokenizer_paths, strict=False):
        tokenizers[name] = _load_tokenizer(path, tokenizer_type)

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
        rt = report.get("round_trip", {}).get(name, {})
        summary_lines.append(
            f"  {name}: {agg.get('record_count', 0)} records, "
            f"{agg.get('token_count', 0)} tokens, "
            f"fertility={agg.get('micro_fertility', '?'):.4f}, "
            f"unknown={agg.get('unknown_token_count', 0)}, "
            f"required-pass={rt.get('required_pass_rate', '?'):.2%}"
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

    detailed_path = args.detailed_records

    if detailed_path is not None and detailed_path.resolve() == output_path.resolve():
        msg = "output-report and detailed-records paths must differ"
        raise ValueError(msg)

    tmp_report = output_path.with_name(f".{output_path.name}.{secrets.token_hex(8)}.verify.tmp")
    tmp_detailed: Path | None = None
    if detailed_path is not None:
        tmp_detailed = detailed_path.with_name(
            f".{detailed_path.name}.{secrets.token_hex(8)}.verify.tmp"
        )

    paths_created: list[Path] = []

    try:
        tmp_report.write_text(report_json, encoding="utf-8")
        loaded = json.loads(tmp_report.read_text(encoding="utf-8"))
        if loaded.get("report_sha256") != report.get("report_sha256"):
            msg = "report verification failed: hash mismatch"
            raise RuntimeError(msg)
        verified_report = tmp_report.read_bytes()

        if tmp_detailed is not None:
            detailed_jsonl = _serialize_detailed(eval_engine)
            tmp_detailed.write_text(detailed_jsonl, encoding="utf-8")
            verified_detailed = tmp_detailed.read_bytes()

        try:
            with open(output_path, "xb") as f:
                f.write(verified_report)
                f.flush()
                os.fsync(f.fileno())
            paths_created.append(output_path)
        except FileExistsError:
            msg = f"refusing to overwrite existing file: {output_path}"
            raise FileExistsError(msg) from None

        if tmp_detailed is not None:
            try:
                with open(detailed_path, "xb") as f:
                    f.write(verified_detailed)
                    f.flush()
                    os.fsync(f.fileno())
                paths_created.append(detailed_path)
            except FileExistsError:
                msg = f"refusing to overwrite existing file: {detailed_path}"
                raise FileExistsError(msg) from None

        final_report = output_path.read_bytes()
        if final_report != verified_report:
            msg = f"byte-verification failed after final write to {output_path}"
            raise RuntimeError(msg)

        if detailed_path is not None:
            final_detailed = detailed_path.read_bytes()
            if verified_detailed is None or final_detailed != verified_detailed:
                msg = f"byte-verification failed after final write to {detailed_path}"
                raise RuntimeError(msg)
    except BaseException:
        for p in paths_created:
            if p.exists():
                p.unlink()
        raise
    finally:
        if tmp_report.exists():
            tmp_report.unlink()
        if tmp_detailed is not None and tmp_detailed.exists():
            tmp_detailed.unlink()

    success = {
        "status": "ok",
        "output": str(output_path),
        "report_sha256": report.get("report_sha256"),
    }
    print(json.dumps(success, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _serialize_detailed(eval_engine: TokenizerEvaluation) -> str:
    records = eval_engine.get_detailed_records()
    lines = [
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) for r in records
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
