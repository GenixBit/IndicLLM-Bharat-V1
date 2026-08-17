from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bharat.eval.local_inference import LocalInferenceConfig, load_local_causal_lm_adapter
from bharat.eval.reporting import compute_aggregate_scores
from bharat.eval.runner import BharatBenchRunner
from bharat.eval.schema import EvalExample, EvalPrediction
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.posttraining.dpo import DPOConfig, dpo_train
from bharat.posttraining.sft import SFTConfig, sft_train
from bharat.tokenizer import BharatTokenizer, load_tokenizer
from bharat.training.pretrain import PretrainConfig, pretrain


@dataclass
class PretrainStageConfig:
    enabled: bool = True
    model_config_path: str = ""
    data_path: str = ""
    val_data_path: str = ""
    max_iters: int = 1000
    batch_size: int = 4
    seq_len: int = 2048
    learning_rate: float = 3e-4
    warmup_iters: int = 100
    gradient_accumulation_steps: int = 1
    device: str = "cpu"
    dtype: str = "float32"
    synthetic_data: bool = False


@dataclass
class SFTStageConfig:
    enabled: bool = True
    data_path: str = ""
    max_iters: int = 200
    batch_size: int = 2
    block_size: int = 1024
    learning_rate: float = 1e-4
    warmup_iters: int = 20
    template_name: str = "indic_instruction"
    device: str = "cpu"


@dataclass
class DPOStageConfig:
    enabled: bool = True
    data_path: str = ""
    max_iters: int = 100
    batch_size: int = 2
    block_size: int = 1024
    learning_rate: float = 5e-5
    beta: float = 0.1
    template_name: str = "indic_instruction"
    device: str = "cpu"


@dataclass
class EvalStageConfig:
    enabled: bool = True
    examples_path: str = ""
    max_new_tokens: int = 64
    device: str = "cpu"


@dataclass
class PipelineConfig:
    name: str
    output_dir: str
    tokenizer_path: str
    pretrain: PretrainStageConfig = field(default_factory=PretrainStageConfig)
    sft: SFTStageConfig = field(default_factory=SFTStageConfig)
    dpo: DPOStageConfig = field(default_factory=DPOStageConfig)
    eval: EvalStageConfig = field(default_factory=EvalStageConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Pipeline config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Pipeline YAML root must be a dict, got {type(data)}")

        pretrain_dict = data.get("pretrain", {})
        sft_dict = data.get("sft", {})
        dpo_dict = data.get("dpo", {})
        eval_dict = data.get("eval", {})

        return cls(
            name=data.get("name", "bharat_pipeline"),
            output_dir=data.get("output_dir", "output/pipeline"),
            tokenizer_path=data.get("tokenizer_path", ""),
            pretrain=PretrainStageConfig(**pretrain_dict),
            sft=SFTStageConfig(**sft_dict),
            dpo=DPOStageConfig(**dpo_dict),
            eval=EvalStageConfig(**eval_dict),
            seed=int(data.get("seed", 42)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


@dataclass
class PipelineResult:
    pipeline_name: str
    completed_stages: list[str]
    pretrain_checkpoint: str | None = None
    sft_checkpoint: str | None = None
    dpo_checkpoint: str | None = None
    final_checkpoint: str | None = None
    pretrain_loss: float | None = None
    sft_loss: float | None = None
    dpo_loss: float | None = None
    eval_scores: dict[str, float] = field(default_factory=dict)
    total_duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_pipeline(
    config: PipelineConfig,
    tokenizer: BharatTokenizer | None = None,
) -> PipelineResult:
    start_time = time.time()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_stages: list[str] = []
    current_model: BharatForCausalLM | None = None
    current_ckpt_path: str | None = None

    pretrain_loss: float | None = None
    sft_loss: float | None = None
    dpo_loss: float | None = None
    eval_scores: dict[str, float] = {}

    pretrain_ckpt_path: str | None = None
    sft_ckpt_path: str | None = None
    dpo_ckpt_path: str | None = None

    if tokenizer is None and config.tokenizer_path:
        tokenizer = load_tokenizer(config.tokenizer_path)

    # ─────────────────────────────────────────────────────────────
    # Stage 1: Pretraining
    # ─────────────────────────────────────────────────────────────
    if config.pretrain.enabled:
        pt_cfg = config.pretrain
        pt_out_dir = out_dir / "pretrain"
        pt_out_dir.mkdir(parents=True, exist_ok=True)

        if pt_cfg.model_config_path:
            with open(pt_cfg.model_config_path, encoding="utf-8") as f:
                model_cfg_dict = yaml.safe_load(f)
            model_config = BharatModelConfig.from_dict(model_cfg_dict)
        else:
            vocab_size = tokenizer.vocab_size if tokenizer else 256
            model_config = BharatModelConfig(
                vocab_size=vocab_size,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )

        pretrain_run_cfg = PretrainConfig(
            model_config=model_config,
            data_path=pt_cfg.data_path,
            val_data_path=pt_cfg.val_data_path or None,
            synthetic_data=pt_cfg.synthetic_data,
            output_dir=str(pt_out_dir),
            max_iters=pt_cfg.max_iters,
            batch_size=pt_cfg.batch_size,
            seq_len=pt_cfg.seq_len,
            learning_rate=pt_cfg.learning_rate,
            warmup_iters=pt_cfg.warmup_iters,
            gradient_accumulation_steps=pt_cfg.gradient_accumulation_steps,
            device=pt_cfg.device,
            dtype=pt_cfg.dtype,
            seed=config.seed,
            save_interval=pt_cfg.max_iters,
        )

        pt_result = pretrain(pretrain_run_cfg, tokenizer=tokenizer)
        pretrain_loss = pt_result.final_loss
        pretrain_ckpt_path = pt_result.checkpoint_path or str(pt_out_dir / "final.pt")
        current_ckpt_path = pretrain_ckpt_path
        completed_stages.append("pretrain")

        current_model = BharatForCausalLM(model_config)
        import torch

        if pretrain_ckpt_path and Path(pretrain_ckpt_path).is_file():
            ckpt_data = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
            current_model.load_state_dict(ckpt_data["model"])

    # ─────────────────────────────────────────────────────────────
    # Stage 2: Supervised Fine-Tuning (SFT)
    # ─────────────────────────────────────────────────────────────
    if config.sft.enabled and config.sft.data_path:
        sft_stage_cfg = config.sft
        sft_out_dir = out_dir / "sft"
        sft_out_dir.mkdir(parents=True, exist_ok=True)

        if current_model is None:
            raise ValueError("SFT stage requires a pretrained model or initial checkpoint")

        sft_run_cfg = SFTConfig(
            data_path=sft_stage_cfg.data_path,
            output_dir=sft_out_dir,
            max_iters=sft_stage_cfg.max_iters,
            batch_size=sft_stage_cfg.batch_size,
            block_size=sft_stage_cfg.block_size,
            learning_rate=sft_stage_cfg.learning_rate,
            warmup_iters=sft_stage_cfg.warmup_iters,
            template_name=sft_stage_cfg.template_name,
            device=sft_stage_cfg.device,
            seed=config.seed,
            save_interval=sft_stage_cfg.max_iters,
        )

        sft_result = sft_train(
            model=current_model,
            config=sft_run_cfg,
            tokenizer=tokenizer,
        )
        sft_loss = sft_result.final_loss
        sft_ckpt_path = str(sft_out_dir / "final.pt")
        current_ckpt_path = sft_ckpt_path
        completed_stages.append("sft")

    # ─────────────────────────────────────────────────────────────
    # Stage 3: Direct Preference Optimization (DPO)
    # ─────────────────────────────────────────────────────────────
    if config.dpo.enabled and config.dpo.data_path:
        dpo_stage_cfg = config.dpo
        dpo_out_dir = out_dir / "dpo"
        dpo_out_dir.mkdir(parents=True, exist_ok=True)

        if current_model is None:
            raise ValueError("DPO stage requires an initialized policy model")

        import copy

        ref_model = copy.deepcopy(current_model)

        dpo_run_cfg = DPOConfig(
            data_path=dpo_stage_cfg.data_path,
            output_dir=dpo_out_dir,
            max_iters=dpo_stage_cfg.max_iters,
            batch_size=dpo_stage_cfg.batch_size,
            block_size=dpo_stage_cfg.block_size,
            learning_rate=dpo_stage_cfg.learning_rate,
            beta=dpo_stage_cfg.beta,
            template_name=dpo_stage_cfg.template_name,
            device=dpo_stage_cfg.device,
            seed=config.seed,
            save_interval=dpo_stage_cfg.max_iters,
        )

        dpo_result = dpo_train(
            policy_model=current_model,
            ref_model=ref_model,
            config=dpo_run_cfg,
            tokenizer=tokenizer,
        )
        dpo_loss = dpo_result.final_loss
        dpo_ckpt_path = str(dpo_out_dir / "final.pt")
        current_ckpt_path = dpo_ckpt_path
        completed_stages.append("dpo")

    # ─────────────────────────────────────────────────────────────
    # Stage 4: BharatBench Evaluation
    # ─────────────────────────────────────────────────────────────
    if config.eval.enabled and config.eval.examples_path and current_ckpt_path:
        eval_cfg = config.eval
        inf_cfg = LocalInferenceConfig(
            checkpoint=current_ckpt_path,
            tokenizer=config.tokenizer_path,
            max_new_tokens=eval_cfg.max_new_tokens,
            device=eval_cfg.device,
        )
        adapter = load_local_causal_lm_adapter(inf_cfg)

        examples: list[EvalExample] = []
        with open(eval_cfg.examples_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        predictions: list[EvalPrediction] = []
        for ex in examples:
            pred_text = adapter.predict(ex)
            predictions.append(EvalPrediction(example_id=ex.example_id, prediction=pred_text))

        runner = BharatBenchRunner()
        results = runner.run(examples, predictions)
        eval_scores = compute_aggregate_scores(results)
        completed_stages.append("eval")

    total_duration = time.time() - start_time

    result = PipelineResult(
        pipeline_name=config.name,
        completed_stages=completed_stages,
        pretrain_checkpoint=pretrain_ckpt_path,
        sft_checkpoint=sft_ckpt_path,
        dpo_checkpoint=dpo_ckpt_path,
        final_checkpoint=current_ckpt_path,
        pretrain_loss=pretrain_loss,
        sft_loss=sft_loss,
        dpo_loss=dpo_loss,
        eval_scores=eval_scores,
        total_duration_sec=round(total_duration, 2),
    )

    summary_file = out_dir / "pipeline_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    return result
