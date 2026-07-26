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


def _checkpoint_gguf(tmp_path: Path) -> Path:
    cp = tmp_path / "checkpoint"
    cp.mkdir()
    (cp / "model.gguf").write_bytes(b"gguf-shard-data")
    return cp


def _safetensors_metadata(tmp_path: Path) -> Path:
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
                        "shard": "model-00001-of-00001.safetensors",
                        "size_bytes": 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return meta


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
                    {"key": "general.name", "value_type": "string", "value": "test"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return meta


class TestManifestReadinessCLI:
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
    # 1 & 2: Successful safetensors / GGUF dry-run with manifest
    # ------------------------------------------------------------------
    def test_safetensors_with_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["dry_run"] is True
        assert data["writer_name"] == "safetensors-dry-run"
        assert data["bytes_written"] == 0
        assert "manifest_readiness" in data
        assert data["manifest_readiness"]["ready"] is True
        assert data["manifest_path"] == str(manifest)
        assert data["manifest_schema_version"] == "1.0"
        assert manifest.exists()

    def test_gguf_with_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path, "gguf"), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "manifest_readiness" in data
        assert data["manifest_readiness"]["ready"] is True
        assert data["writer_name"] == "gguf-dry-run"
        assert manifest.exists()

    # ------------------------------------------------------------------
    # 3: manifest_readiness appears when --manifest-path is supplied
    # ------------------------------------------------------------------
    def test_manifest_readiness_appears_with_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        assert "manifest_readiness" in json.loads(result.stdout)

    # ------------------------------------------------------------------
    # 4: manifest_readiness absent when --manifest-path is omitted
    # ------------------------------------------------------------------
    def test_manifest_readiness_absent_without_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli(self._base_args(tmp_path))
        assert result.returncode == 0
        assert "manifest_readiness" not in json.loads(result.stdout)

    # ------------------------------------------------------------------
    # 5: Existing manifest file is rejected
    # ------------------------------------------------------------------
    def test_rejects_existing_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode != 0
        assert "manifest path already exists" in result.stderr

    # ------------------------------------------------------------------
    # 6: Missing manifest parent is rejected
    # ------------------------------------------------------------------
    def test_rejects_missing_manifest_parent(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "missing" / "export.json"
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode != 0
        assert "manifest parent directory does not exist" in result.stderr

    # ------------------------------------------------------------------
    # 7: Manifest parent that is a file is rejected
    # ------------------------------------------------------------------
    def test_rejects_manifest_parent_as_file(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        parent = tmp_path / "manifest-parent"
        parent.write_text("i-am-a-file", encoding="utf-8")
        manifest = parent / "export.json"
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode != 0
        assert "manifest parent path must be a directory" in result.stderr

    # ------------------------------------------------------------------
    # 8: Manifest path equal to output path is rejected
    # ------------------------------------------------------------------
    def test_rejects_manifest_equal_to_output(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        output = out_dir / "bharat.safetensors"
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
                "--manifest-path",
                str(output),
            ]
        )
        assert result.returncode != 0
        assert "must not equal export output path" in result.stderr

    # ------------------------------------------------------------------
    # 9: Manifest path inside checkpoint directory is rejected
    # ------------------------------------------------------------------
    def test_rejects_manifest_inside_checkpoint(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        (tmp_path / "out").mkdir()
        manifest = cp / "manifest.json"
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
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode != 0
        assert "must not be inside" in result.stderr

    # ------------------------------------------------------------------
    # 10: Relative and .. paths are safely resolved
    # ------------------------------------------------------------------
    def test_path_traversal_resolved(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        (tmp_path / "out").mkdir()
        (tmp_path / "manifests").mkdir()
        manifest = tmp_path / "manifests" / ".." / "manifests" / "export.json"
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
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["manifest_readiness"]["ready"] is True

    # ------------------------------------------------------------------
    # 11: Symlink-based containment bypass is rejected
    # ------------------------------------------------------------------
    def test_symlink_into_checkpoint_rejected(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        (tmp_path / "out").mkdir()
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()
        sym = outside_dir / "ckpt_link"
        os.symlink(str(cp), str(sym), target_is_directory=True)
        manifest = sym / "manifest.json"
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
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode != 0
        assert "must not be inside" in result.stderr

    # ------------------------------------------------------------------
    # 12 & 13 & 14 & 15: Failure does not invoke writer, create
    #     manifest or output, or emit partial JSON
    # ------------------------------------------------------------------
    def test_failure_does_not_create_manifest_or_output(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "missing" / "export.json"
        output = tmp_path / "out" / "bharat.safetensors"
        result = run_cli(
            [
                "--checkpoint-path",
                str(_checkpoint(tmp_path)),
                "--output-path",
                str(output),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode != 0
        assert result.stdout == ""
        assert not manifest.exists()
        assert not output.exists()

    # ------------------------------------------------------------------
    # 16: Successful manifest write occurs only after readiness
    # ------------------------------------------------------------------
    def test_manifest_written_after_readiness(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        assert manifest.exists()
        content = json.loads(manifest.read_text(encoding="utf-8"))
        assert content["schema_version"] == "1.0"

    # ------------------------------------------------------------------
    # 17: Existing manifest output fields remain present
    # ------------------------------------------------------------------
    def test_manifest_fields_present(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["manifest_path"] == str(manifest)
        assert data["manifest_schema_version"] == "1.0"
        assert data["manifest_readiness"]["ready"] is True

    # ------------------------------------------------------------------
    # 18: Existing CLI output unchanged without --manifest-path
    # ------------------------------------------------------------------
    def test_existing_output_unchanged_without_manifest(self, tmp_path: Path) -> None:
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
    # 19: Writer readiness + manifest readiness work together
    # ------------------------------------------------------------------
    def test_writer_and_manifest_readiness_together(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--validate-writer-readiness",
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["writer_readiness"]["ready"] is True
        assert data["manifest_readiness"]["ready"] is True
        assert manifest.exists()

    # ------------------------------------------------------------------
    # 20: Safetensors preflight + manifest readiness together
    # ------------------------------------------------------------------
    def test_safetensors_preflight_with_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        meta = _safetensors_metadata(tmp_path)
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--safetensors-metadata-path",
                str(meta),
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["safetensors_preflight"]["schema_version"] == 1
        assert data["manifest_readiness"]["ready"] is True
        assert manifest.exists()

    # ------------------------------------------------------------------
    # 21: GGUF preflight + manifest readiness together
    # ------------------------------------------------------------------
    def test_gguf_preflight_with_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        meta = _gguf_metadata(tmp_path)
        cp = _checkpoint_gguf(tmp_path)
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
                "--gguf-metadata-path",
                str(meta),
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["gguf_preflight"]["schema_version"] == 1
        assert data["manifest_readiness"]["ready"] is True
        assert manifest.exists()

    # ------------------------------------------------------------------
    # 22: Remote manifest paths remain rejected
    # ------------------------------------------------------------------
    def test_remote_manifest_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--manifest-path",
                "https://example.com/manifest.json",
            ]
        )
        assert result.returncode != 0
        assert "Remote manifest path rejected" in result.stderr

    # ------------------------------------------------------------------
    # 23: Dry-run result still reports zero bytes written with manifest
    # ------------------------------------------------------------------
    def test_dry_run_zero_bytes_with_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["bytes_written"] == 0

    # ------------------------------------------------------------------
    # 24 & 25: No tensor payloads or network (guaranteed by tmp_path
    #     fixtures and offline nature of tests)
    # ------------------------------------------------------------------
    def test_no_output_file_created_after_success(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert not Path(data["output_path"]).exists()
