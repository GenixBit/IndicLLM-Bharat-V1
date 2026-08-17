import os
import subprocess
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from bharat.tokenizer import BharatTokenizer, tokenizer_hash


@dataclass
class CheckpointMetadata:
    tokenizer_type: str = ""
    tokenizer_hash: str = ""
    vocab_size: int = 0
    git_sha: str = ""
    data_version: str = ""
    seed: int = 0
    training_step: int = 0
    torch_version: str = ""
    package_versions: dict[str, str] = field(default_factory=dict)


def get_git_sha(cwd: str | Path | None = None) -> str:
    repo_root = Path(__file__).resolve().parent.parent.parent
    target_cwd = Path(cwd) if cwd else repo_root
    try:
        env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(target_cwd),
            env=env,
            timeout=5,
        )
        if result.returncode == 0 and len(result.stdout.strip()) == 40:
            return result.stdout.strip()
    except Exception:
        pass

    try:
        git_dir = target_cwd / ".git"
        if git_dir.is_file():
            content = git_dir.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                git_dir = (target_cwd / content.split(":", 1)[1].strip()).resolve()
        if git_dir.is_dir():
            head_file = git_dir / "HEAD"
            if head_file.is_file():
                head_content = head_file.read_text(encoding="utf-8").strip()
                if head_content.startswith("ref:"):
                    ref_path = head_content.split(":", 1)[1].strip()
                    ref_file = git_dir / ref_path
                    if ref_file.is_file():
                        candidate = ref_file.read_text(encoding="utf-8").strip()
                        if len(candidate) == 40:
                            return candidate
                    packed = git_dir / "packed-refs"
                    if packed.is_file():
                        for line in packed.read_text(encoding="utf-8").splitlines():
                            if line and not line.startswith("#") and not line.startswith("^"):
                                parts = line.strip().split()
                                if len(parts) == 2 and parts[1] == ref_path and len(parts[0]) == 40:
                                    return parts[0]
                elif len(head_content) == 40:
                    return head_content
    except Exception:
        pass

    return ""


def get_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ["torch", "transformers", "tokenizers", "numpy", "datasets"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def make_checkpoint_data(
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    tokenizer: BharatTokenizer | None = None,
    step: int = 0,
    seed: int = 0,
    data_version: str = "",
    loss: float | None = None,
) -> dict[str, Any]:
    ckpt: dict[str, Any] = {
        "model": model_state,
    }

    if optimizer_state is not None:
        ckpt["optimizer"] = optimizer_state
    if scheduler_state is not None:
        ckpt["scheduler"] = scheduler_state
    if config is not None:
        ckpt["config"] = config
    if loss is not None:
        ckpt["loss"] = loss

    meta = CheckpointMetadata(
        git_sha=get_git_sha(),
        data_version=data_version,
        seed=seed,
        training_step=step,
        torch_version=torch.__version__,
        package_versions=get_package_versions(),
    )

    if tokenizer is not None:
        meta.tokenizer_type = tokenizer.tokenizer_type
        meta.tokenizer_hash = tokenizer_hash(tokenizer)
        meta.vocab_size = tokenizer.vocab_size

    ckpt["metadata"] = asdict(meta)
    ckpt["rng_state"] = _capture_rng_state()

    return ckpt


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": None,
        "torch": None,
        "cuda": {},
    }
    with suppress(Exception):
        import random

        state["python"] = random.getstate()

    with suppress(Exception):
        state["torch"] = torch.get_rng_state().tolist()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with suppress(Exception):
                state["cuda"][str(i)] = torch.cuda.get_rng_state(i).tolist()

    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if state.get("python"):
        import random

        random.setstate(state["python"])

    if state.get("torch"):
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))

    cuda_state = state.get("cuda", {})
    if torch.cuda.is_available():
        for device_idx, device_state in cuda_state.items():
            with suppress(Exception):
                torch.cuda.set_rng_state(
                    torch.tensor(device_state, dtype=torch.uint8),
                    device=int(device_idx),
                )


def validate_checkpoint(
    ckpt: dict[str, Any],
    tokenizer: BharatTokenizer | None = None,
) -> CheckpointMetadata:
    meta_dict = ckpt.get("metadata")
    if not meta_dict:
        raise ValueError(
            "Checkpoint has no metadata section. Use the new training code to create checkpoints."
        )

    meta = CheckpointMetadata(**meta_dict)

    if tokenizer is not None and meta.tokenizer_hash:
        current_hash = tokenizer_hash(tokenizer)
        if current_hash != meta.tokenizer_hash:
            raise ValueError(
                f"Tokenizer mismatch: checkpoint has {meta.tokenizer_type} "
                f"(hash={meta.tokenizer_hash[:12]}...), "
                f"current tokenizer is {tokenizer.tokenizer_type} "
                f"(hash={current_hash[:12]}...). "
                f"Use the same tokenizer that was used during training."
            )

    if tokenizer is not None and meta.vocab_size and meta.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Vocab size mismatch: checkpoint has {meta.vocab_size}, "
            f"current tokenizer has {tokenizer.vocab_size}"
        )

    return meta


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
    tokenizer: BharatTokenizer | None = None,
    step: int = 0,
    seed: int = 0,
    data_version: str = "",
    loss: float | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = make_checkpoint_data(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict() if optimizer else None,
        scheduler_state=scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,  # type: ignore[union-attr]
        config=config,
        tokenizer=tokenizer,
        step=step,
        seed=seed,
        data_version=data_version,
        loss=loss,
    )

    torch.save(ckpt, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    tokenizer: BharatTokenizer | None = None,
    device: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    validate_checkpoint(ckpt, tokenizer=tokenizer)

    model.load_state_dict(ckpt["model"], strict=strict)

    result: dict[str, Any] = {}
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if "rng_state" in ckpt:
        _restore_rng_state(ckpt["rng_state"])

    meta = CheckpointMetadata(**ckpt.get("metadata", {}))
    result["metadata"] = meta
    result["step"] = meta.training_step
    result["config"] = ckpt.get("config", {})
    result["loss"] = ckpt.get("loss", None)

    return result


def save_checkpoint_for_legacy(
    path: str | Path,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    tokenizer: BharatTokenizer | None = None,
    step: int = 0,
) -> Path:
    """Backward-compatible save that works with both legacy old-style loaders and new loaders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: dict[str, Any] = {"model": model_state}

    if optimizer_state is not None:
        ckpt["optimizer"] = optimizer_state
    if config is not None:
        ckpt["config"] = config
    ckpt["step"] = step

    meta = CheckpointMetadata(
        git_sha=get_git_sha(),
        training_step=step,
        torch_version=torch.__version__,
        package_versions=get_package_versions(),
    )
    if tokenizer is not None:
        meta.tokenizer_type = tokenizer.tokenizer_type
        meta.tokenizer_hash = tokenizer_hash(tokenizer)
        meta.vocab_size = tokenizer.vocab_size

    ckpt["metadata"] = asdict(meta)

    torch.save(ckpt, path)
    return path
