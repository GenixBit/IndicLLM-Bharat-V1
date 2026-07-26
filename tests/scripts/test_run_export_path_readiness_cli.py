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


class TestExportPathReadinessCLI:
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
    # 1 & 2: Successful safetensors / GGUF metadata path readiness
    # ------------------------------------------------------------------
    def test_safetensors_path_readiness(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        meta = _safetensors_metadata(tmp_path)
        result = run_cli([*self._base_args(tmp_path), "--safetensors-metadata-path", str(meta)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "export_path_readiness" in data
        assert data["export_path_readiness"]["ready"] is True
        assert data["safetensors_preflight"]["schema_version"] == 1

    def test_gguf_path_readiness(self, tmp_path: Path) -> None:
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
                "--gguf-metadata-path",
                str(meta),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "export_path_readiness" in data
        assert data["export_path_readiness"]["ready"] is True
        assert data["gguf_preflight"]["schema_version"] == 1

    # ------------------------------------------------------------------
    # 3: Successful manifest-only path readiness
    # ------------------------------------------------------------------
    def test_manifest_only_path_readiness(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "export_path_readiness" in data
        assert data["export_path_readiness"]["ready"] is True
        assert data["export_path_readiness"]["metadata_paths"] == []

    # ------------------------------------------------------------------
    # 4 & 5: Manifest plus metadata path readiness
    # ------------------------------------------------------------------
    def test_manifest_plus_safetensors_path_readiness(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        meta = _safetensors_metadata(tmp_path)
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--manifest-path",
                str(manifest),
                "--safetensors-metadata-path",
                str(meta),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["export_path_readiness"]["ready"] is True
        assert data["export_path_readiness"]["manifest_path"] is not None
        assert len(data["export_path_readiness"]["metadata_paths"]) == 1

    def test_manifest_plus_gguf_path_readiness(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
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
                "--manifest-path",
                str(manifest),
                "--gguf-metadata-path",
                str(meta),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["export_path_readiness"]["ready"] is True

    # ------------------------------------------------------------------
    # 6 & 7: Presence conditions
    # ------------------------------------------------------------------
    def test_path_readiness_appears_with_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        meta = _safetensors_metadata(tmp_path)
        result = run_cli([*self._base_args(tmp_path), "--safetensors-metadata-path", str(meta)])
        assert result.returncode == 0
        assert "export_path_readiness" in json.loads(result.stdout)

    def test_path_readiness_appears_with_manifest_only(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--manifest-path", str(manifest)])
        assert result.returncode == 0
        assert "export_path_readiness" in json.loads(result.stdout)

    # ------------------------------------------------------------------
    # 8: Absent when no manifest or metadata path
    # ------------------------------------------------------------------
    def test_path_readiness_absent_without_targets(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli(self._base_args(tmp_path))
        assert result.returncode == 0
        assert "export_path_readiness" not in json.loads(result.stdout)

    # ------------------------------------------------------------------
    # 9: Resolved metadata paths are deterministic
    # ------------------------------------------------------------------
    def test_deterministic_metadata_paths(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        meta = _safetensors_metadata(tmp_path)
        result = run_cli([*self._base_args(tmp_path), "--safetensors-metadata-path", str(meta)])
        data = json.loads(result.stdout)
        paths = data["export_path_readiness"]["metadata_paths"]
        assert len(paths) == 1
        assert paths == sorted(paths)

    # ------------------------------------------------------------------
    # 10: Missing metadata file rejected
    # ------------------------------------------------------------------
    def test_rejects_missing_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        missing = tmp_path / "missing.json"
        result = run_cli([*self._base_args(tmp_path), "--safetensors-metadata-path", str(missing)])
        assert result.returncode != 0
        assert "does not exist" in result.stderr

    # ------------------------------------------------------------------
    # 11: Metadata directory rejected
    # ------------------------------------------------------------------
    def test_rejects_metadata_directory(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        d = tmp_path / "adir"
        d.mkdir()
        result = run_cli([*self._base_args(tmp_path), "--safetensors-metadata-path", str(d)])
        assert result.returncode != 0
        assert "must be a file" in result.stderr

    # ------------------------------------------------------------------
    # 12: Metadata equal to output rejected
    # ------------------------------------------------------------------
    def test_rejects_metadata_equal_output(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        output = out_dir / "bharat.safetensors"
        output.write_text("collision", encoding="utf-8")
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
                "--safetensors-metadata-path",
                str(output),
            ]
        )
        assert result.returncode != 0
        assert "must not equal export output path" in result.stderr

    # ------------------------------------------------------------------
    # 13: Metadata equal to manifest rejected
    # ------------------------------------------------------------------
    def test_rejects_metadata_equal_manifest(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "shared.json"
        manifest.write_text("{}", encoding="utf-8")
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
                "--safetensors-metadata-path",
                str(manifest),
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode != 0
        assert "must not equal export manifest path" in result.stderr

    # ------------------------------------------------------------------
    # 14 & 15: Symlink resolution for output / manifest collision
    # ------------------------------------------------------------------
    def test_rejects_symlink_metadata_to_output(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        output = out_dir / "bharat.safetensors"
        output.write_text("real-output", encoding="utf-8")
        link = tmp_path / "meta_link.json"
        os.symlink(str(output), str(link))
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
                "--safetensors-metadata-path",
                str(link),
            ]
        )
        assert result.returncode != 0
        assert "must not equal export output path" in result.stderr

    def test_rejects_symlink_metadata_to_manifest(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "target.json"
        manifest.write_text("{}", encoding="utf-8")
        link = tmp_path / "meta_link.json"
        os.symlink(str(manifest), str(link))
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
                "--safetensors-metadata-path",
                str(link),
                "--manifest-path",
                str(manifest),
            ]
        )
        assert result.returncode != 0
        assert "must not equal export manifest path" in result.stderr

    # ------------------------------------------------------------------
    # 16: Relative and .. paths resolved safely
    # ------------------------------------------------------------------
    def test_path_traversal_resolved(self, tmp_path: Path) -> None:
        cp = _checkpoint(tmp_path)
        (tmp_path / "out").mkdir()
        _safetensors_metadata(tmp_path)
        meta_via_dotdot = tmp_path / "out" / ".." / "safetensors-meta.json"
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
                "--safetensors-metadata-path",
                str(meta_via_dotdot),
            ]
        )
        assert result.returncode == 0

    # ------------------------------------------------------------------
    # 17-24: Failure ordering guarantees
    # ------------------------------------------------------------------
    def test_failure_blocks_downstream(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        missing = tmp_path / "missing.json"
        manifest = tmp_path / "manifests" / "export.json"
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--validate-writer-readiness",
                "--manifest-path",
                str(manifest),
                "--safetensors-metadata-path",
                str(missing),
            ]
        )
        assert result.returncode != 0
        assert result.stdout == ""
        assert not manifest.exists()
        assert "does not exist" in result.stderr

    # ------------------------------------------------------------------
    # 25: Existing CLI unchanged when readiness does not run
    # ------------------------------------------------------------------
    def test_existing_output_unchanged_without_targets(self, tmp_path: Path) -> None:
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
    # 26: Existing safetensors preflight remains correct
    # ------------------------------------------------------------------
    def test_safetensors_preflight_preserved(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        meta = _safetensors_metadata(tmp_path)
        result = run_cli([*self._base_args(tmp_path), "--safetensors-metadata-path", str(meta)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["safetensors_preflight"]["schema_version"] == 1

    # ------------------------------------------------------------------
    # 27: Existing GGUF preflight remains correct
    # ------------------------------------------------------------------
    def test_gguf_preflight_preserved(self, tmp_path: Path) -> None:
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
                "--gguf-metadata-path",
                str(meta),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["gguf_preflight"]["schema_version"] == 1

    # ------------------------------------------------------------------
    # 28: Existing writer_readiness preserved
    # ------------------------------------------------------------------
    def test_writer_readiness_preserved(self, tmp_path: Path) -> None:
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
        assert data["writer_readiness"]["ready"] is True

    # ------------------------------------------------------------------
    # 29: Existing manifest_readiness preserved
    # ------------------------------------------------------------------
    def test_manifest_readiness_preserved(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        manifest = tmp_path / "manifests" / "export.json"
        manifest.parent.mkdir()
        meta = _safetensors_metadata(tmp_path)
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--manifest-path",
                str(manifest),
                "--safetensors-metadata-path",
                str(meta),
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["manifest_readiness"]["ready"] is True

    # ------------------------------------------------------------------
    # 30: Remote metadata and manifest paths remain rejected
    # ------------------------------------------------------------------
    def test_remote_metadata_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        result = run_cli(
            [
                *self._base_args(tmp_path),
                "--safetensors-metadata-path",
                "https://example.com/meta.json",
            ]
        )
        assert result.returncode != 0
        assert "Remote safetensors metadata path rejected" in result.stderr
