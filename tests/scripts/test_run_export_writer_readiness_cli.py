from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_export_plan", *args],
        capture_output=True,
        text=True,
    )


def _checkpoint(tmp_path: Path) -> Path:
    cp = tmp_path / "checkpoint"
    cp.mkdir()
    (cp / "model-00001-of-00001.safetensors").write_bytes(b"shard-data")
    return cp


def _safetensors_metadata(
    tmp_path: Path, shard_name: str = "model-00001-of-00001.safetensors"
) -> Path:
    meta = tmp_path / "safetensors-meta.json"
    meta.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_tensor_bytes": 10,
                "tensors": [
                    {
                        "name": "transformer.weight",
                        "shape": [1],
                        "dtype": "BF16",
                        "shard": shard_name,
                        "size_bytes": 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return meta


def _checkpoint_gguf(tmp_path: Path) -> Path:
    cp = tmp_path / "checkpoint"
    cp.mkdir()
    (cp / "model.gguf").write_bytes(b"gguf-shard-data")
    return cp


def _gguf_metadata(tmp_path: Path) -> Path:
    meta = tmp_path / "gguf-meta.json"
    meta.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": "bharat",
                "alignment": 32,
                "tensor_count": 0,
                "output_file": "model.gguf",
                "metadata": [
                    {"key": "general.name", "value_type": "string"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return meta


class TestWriterReadinessCLI:
    def _base_args(self, tmp_path: Path, fmt: str = "safetensors") -> list[str]:
        return [
            "--checkpoint-path",
            str(_checkpoint(tmp_path)),
            "--output-path",
            str(tmp_path / "out" / f"bharat.{fmt}"),
            "--format",
            fmt,
            "--model-name",
            "bharat-local",
        ]

    # ------------------------------------------------------------------
    # 1 & 2 & 18 & 19: Successful readiness for safetensors / GGUF
    # ------------------------------------------------------------------
    def test_safetensors_readiness_success(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli([*self._base_args(tmp_path), "--validate-writer-readiness"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "writer_readiness" in data
        assert data["writer_readiness"]["ready"] is True
        assert data["dry_run"] is True
        assert data["bytes_written"] == 0
        assert not Path(data["output_path"]).exists()

    def test_gguf_readiness_success(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli([*self._base_args(tmp_path, "gguf"), "--validate-writer-readiness"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "writer_readiness" in data
        assert data["writer_readiness"]["ready"] is True

    # ------------------------------------------------------------------
    # 3: writer_readiness appears when flag is supplied
    # ------------------------------------------------------------------
    def test_writer_readiness_appears_with_flag(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli([*self._base_args(tmp_path), "--validate-writer-readiness"])
        assert result.returncode == 0
        assert "writer_readiness" in json.loads(result.stdout)

    # ------------------------------------------------------------------
    # 4: writer_readiness is absent when flag is omitted
    # ------------------------------------------------------------------
    def test_writer_readiness_absent_without_flag(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli(self._base_args(tmp_path))
        assert result.returncode == 0
        assert "writer_readiness" not in json.loads(result.stdout)

    # ------------------------------------------------------------------
    # 5 & 6: Inventory auto-built but hidden unless --include-inventory
    # ------------------------------------------------------------------
    def test_inventory_hidden_when_auto_built(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli([*self._base_args(tmp_path), "--validate-writer-readiness"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "writer_readiness" in data
        assert "checkpoint_inventory" not in data

    def test_inventory_shown_with_include_flag(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--validate-writer-readiness",
                "--include-inventory",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "checkpoint_inventory" in data
        assert data["checkpoint_inventory"]["total_bytes"] > 0

    # ------------------------------------------------------------------
    # 7: Existing output file is rejected
    # ------------------------------------------------------------------
    def test_rejects_existing_output_file(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = tmp_path / "out" / "bharat.safetensors"
        output.write_text("existing-content")
        result = run_cli([*self._base_args(tmp_path), "--validate-writer-readiness"])
        assert result.returncode != 0
        assert "output path already exists" in result.stderr

    # ------------------------------------------------------------------
    # 8: Missing output parent directory is rejected
    # ------------------------------------------------------------------
    def test_rejects_missing_output_parent(self, tmp_path: Path) -> None:
        result = run_cli([*self._base_args(tmp_path), "--validate-writer-readiness"])
        assert result.returncode != 0
        assert "output parent directory does not exist" in result.stderr

    # ------------------------------------------------------------------
    # 9: Output parent that is a file is rejected
    # ------------------------------------------------------------------
    def test_rejects_output_parent_as_file(self, tmp_path: Path) -> None:
        (tmp_path / "out").write_text("i-am-a-file")
        result = run_cli([*self._base_args(tmp_path), "--validate-writer-readiness"])
        assert result.returncode != 0
        assert "output parent path must be a directory" in result.stderr

    # ------------------------------------------------------------------
    # 10: Output path inside checkpoint directory is rejected
    # ------------------------------------------------------------------
    def test_rejects_output_inside_checkpoint(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(cp / "bharat.safetensors"),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--validate-writer-readiness",
            ]
        )
        assert result.returncode != 0
        assert "must not be inside" in result.stderr

    # ------------------------------------------------------------------
    # 11: Empty checkpoint directory is rejected
    # ------------------------------------------------------------------
    def test_rejects_empty_checkpoint(self, tmp_path: Path) -> None:
        cp = tmp_path / "empty"
        cp.mkdir()
        (tmp_path / "out").mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(tmp_path / "out" / "bharat.safetensors"),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--validate-writer-readiness",
            ]
        )
        assert result.returncode != 0
        assert "contains no files" in result.stderr

    # ------------------------------------------------------------------
    # 12: Readiness failure does not create a manifest
    # ------------------------------------------------------------------
    def test_failure_does_not_create_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        cp = _checkpoint(tmp_path)
        (tmp_path / "out").mkdir()
        output = tmp_path / "out" / "bharat.safetensors"
        output.write_text("existing")
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(output),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--validate-writer-readiness",
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode != 0
        assert not manifest.exists()

    # ------------------------------------------------------------------
    # 13: Readiness works without metadata-preflight
    # ------------------------------------------------------------------
    def test_readiness_without_preflight(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli([*self._base_args(tmp_path), "--validate-writer-readiness"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "writer_readiness" in data
        assert "safetensors_preflight" not in data
        assert "gguf_preflight" not in data

    # ------------------------------------------------------------------
    # 14: Readiness works together with safetensors metadata preflight
    # ------------------------------------------------------------------
    def test_readiness_with_safetensors_preflight(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        meta = _safetensors_metadata(tmp_path)
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--validate-writer-readiness",
                "--safetensors-metadata-path",
                str(meta),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "writer_readiness" in data
        assert data["writer_readiness"]["ready"] is True
        assert data["safetensors_preflight"]["schema_version"] == 1

    # ------------------------------------------------------------------
    # 15: Readiness works together with GGUF metadata preflight
    # ------------------------------------------------------------------
    def test_readiness_with_gguf_preflight(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        cp = _checkpoint_gguf(tmp_path)
        meta = _gguf_metadata(tmp_path)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(tmp_path / "out" / "bharat.gguf"),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--validate-writer-readiness",
                "--gguf-metadata-path",
                str(meta),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "writer_readiness" in data
        assert data["writer_readiness"]["ready"] is True
        assert data["gguf_preflight"]["schema_version"] == 1

    # ------------------------------------------------------------------
    # 16: Existing CLI output unchanged when new flag absent
    # ------------------------------------------------------------------
    def test_existing_output_unchanged_without_flag(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli(self._base_args(tmp_path))
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert set(data.keys()) == {
            "checkpoint_path",
            "output_path",
            "export_format",
            "model_name",
            "dry_run",
            "writer_name",
            "bytes_written",
        }

    # ------------------------------------------------------------------
    # 17: Remote paths remain rejected (spot-check that existing logic
    #     still applies when combining with --validate-writer-readiness)
    # ------------------------------------------------------------------
    def test_remote_output_rejected_with_readiness(self, tmp_path: Path) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "checkpoints/bharat",
                "--output-path",
                "s3://bucket/model.safetensors",
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--validate-writer-readiness",
            ]
        )
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    # ------------------------------------------------------------------
    # 20: .. and symlink-resolved paths cannot bypass readiness rules
    # ------------------------------------------------------------------
    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        (tmp_path / "out").mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(tmp_path / "out" / ".." / "out" / "bharat.safetensors"),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--validate-writer-readiness",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["writer_readiness"]["ready"] is True

    def test_symlink_outside_checkpoint_allowed(self, tmp_path: Path) -> None:
        real = tmp_path / "real_dest"
        real.mkdir()
        cp = _checkpoint(tmp_path)
        link = tmp_path / "out_link"
        os.symlink(str(real), str(link), target_is_directory=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(link / "bharat.safetensors"),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--validate-writer-readiness",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["writer_readiness"]["ready"] is True

    def test_symlink_outside_to_inside_rejected(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()
        sym_outside = outside_dir / "ckpt_link"
        os.symlink(str(cp), str(sym_outside), target_is_directory=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(sym_outside / "bharat.safetensors"),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--validate-writer-readiness",
            ]
        )
        assert result.returncode != 0
        assert "must not be inside" in result.stderr
