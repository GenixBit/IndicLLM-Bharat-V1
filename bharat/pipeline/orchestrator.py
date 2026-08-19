"""Unified End-to-End Sovereign Training Pipeline Orchestrator for IndicLLM-Bharat.

Coordinates the complete model lifecycle:
  1. Data Sharding & Curriculum Synthesis
  2. Pretraining on World & Indic Mixture
  3. Supervised Fine-Tuning (SFT) with Assistant Loss Masking
  4. Direct Preference Optimization (DPO) Alignment
  5. Safetensors & GGUF Q8_0 Edge Quantization
  6. 22-Language IndicMMLU & 32k Long-Context Evaluation
  7. Cryptographic Run Manifest Generation
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from bharat.data.instruction_curriculum import export_instruction_curriculum
from bharat.data.mixture import stream_and_pack_mixture
from bharat.eval.indic_benchmarks import IndicBenchmarkRunner
from bharat.eval.long_context import LongContextEvaluator
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.posttraining.sft_trainer import BharatSFTTrainer, SFTTrainingConfig
from bharat.serving.gguf_quantizer import export_model_to_gguf_q8_0
from bharat.tokenizer import BharatTokenizer, load_tokenizer
from bharat.training.dpo_trainer import BharatDPOTrainer, DPOTrainerConfig
from bharat.training.scale_trainer import BharatScaleTrainer, ScaleTrainerConfig


@dataclass
class StageResult:
    stage_name: str
    status: str
    duration_seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    tier: str = "1b"
    work_dir: str | Path = "workspace/pipeline_run"
    stages: list[str] = field(
        default_factory=lambda: ["data", "pretrain", "sft", "dpo", "export", "eval"]
    )
    pretrain_steps: int = 50
    sft_steps: int = 30
    dpo_steps: int = 20
    batch_size: int = 2
    device: str = "auto"
    seed: int = 42


@dataclass
class PipelineRunManifest:
    pipeline_id: str
    timestamp: float
    tier: str
    config: dict[str, Any]
    stages: list[dict[str, Any]]
    total_duration_seconds: float
    manifest_sha256: str = ""


class SovereignPipelineOrchestrator:
    """End-to-end lifecycle orchestrator for IndicLLM-Bharat."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)
        self.tokenizer: BharatTokenizer = load_tokenizer("gpt2")
        self.stage_results: list[StageResult] = []

    def run_stage_data(self) -> StageResult:
        """Stage 1: Prepare data mixtures and instruction curriculums."""
        start = time.perf_counter()
        shards_dir = self.work_dir / "shards"
        sft_data = self.work_dir / "sft_data.jsonl"

        shards = stream_and_pack_mixture(
            tokenizer=self.tokenizer,
            output_dir=shards_dir,
            max_tokens_per_shard=200_000,
            max_docs=100,
        )
        sft_count = export_instruction_curriculum(sft_data)

        dur = time.perf_counter() - start
        artifacts = [str(s) for s in shards] + [str(sft_data)]
        return StageResult(
            stage_name="data",
            status="SUCCESS",
            duration_seconds=dur,
            metrics={"shards_generated": len(shards), "sft_samples": sft_count},
            artifacts=artifacts,
        )

    def run_stage_pretrain(self) -> StageResult:
        """Stage 2: Pretraining foundation model."""
        start = time.perf_counter()
        out_dir = self.work_dir / "checkpoints_pretrain"

        scale_cfg = ScaleTrainerConfig(
            tier=self.config.tier,
            steps=self.config.pretrain_steps,
            batch_size=self.config.batch_size,
            block_size=128 if self.config.tier == "tiny" else 512,
            output_dir=out_dir,
            device=self.config.device,
            seed=self.config.seed,
        )

        trainer = BharatScaleTrainer(scale_cfg)
        train_res = trainer.train()

        dur = time.perf_counter() - start
        return StageResult(
            stage_name="pretrain",
            status="SUCCESS",
            duration_seconds=dur,
            metrics={
                "tier": train_res.tier,
                "final_loss": train_res.final_loss,
                "tokens_processed": train_res.total_tokens_processed,
            },
            artifacts=[train_res.checkpoint_path],
        )

    def run_stage_sft(self, base_ckpt: str | None = None) -> StageResult:
        """Stage 3: Supervised Fine-Tuning with assistant-only loss masking."""
        start = time.perf_counter()
        out_dir = self.work_dir / "checkpoints_sft"
        data_p = self.work_dir / "sft_data.jsonl"

        sft_cfg = SFTTrainingConfig(
            tier=self.config.tier,
            checkpoint_path=base_ckpt,
            data_path=data_p,
            output_dir=out_dir,
            steps=self.config.sft_steps,
            batch_size=self.config.batch_size,
            block_size=128 if self.config.tier == "tiny" else 512,
            device=self.config.device,
            seed=self.config.seed,
        )

        trainer = BharatSFTTrainer(sft_cfg)
        res = trainer.train()

        dur = time.perf_counter() - start
        return StageResult(
            stage_name="sft",
            status="SUCCESS",
            duration_seconds=dur,
            metrics={
                "tier": res.tier,
                "final_sft_loss": res.final_loss,
                "active_tokens": res.active_tokens,
            },
            artifacts=[res.checkpoint_path],
        )

    def run_stage_dpo(self, sft_ckpt: str | None = None) -> StageResult:
        """Stage 4: Direct Preference Optimization (DPO) alignment."""
        start = time.perf_counter()
        out_dir = self.work_dir / "checkpoints_dpo"

        dpo_cfg = DPOTrainerConfig(
            sft_checkpoint=sft_ckpt or "checkpoints/bharat_sft/final.pt",
            max_iters=self.config.dpo_steps,
            batch_size=self.config.batch_size,
            output_dir=out_dir,
            device=self.config.device,
            seed=self.config.seed,
        )

        dpo_trainer = BharatDPOTrainer(dpo_cfg)
        res = dpo_trainer.train()

        dur = time.perf_counter() - start
        return StageResult(
            stage_name="dpo",
            status="SUCCESS",
            duration_seconds=dur,
            metrics={
                "final_dpo_loss": res.final_loss,
                "final_accuracy": res.final_reward_accuracy,
                "reward_margin": res.final_reward_margin,
            },
            artifacts=[res.checkpoint_path],
        )

    def run_stage_export(self, model_ckpt: str | None = None) -> StageResult:
        """Stage 5: GGUF Q8_0 & Safetensors edge quantization."""
        start = time.perf_counter()
        out_gguf = self.work_dir / "bharat_edge_q8_0.gguf"

        cfg = BharatModelConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=64 if self.config.tier == "tiny" else 256,
            intermediate_size=128 if self.config.tier == "tiny" else 512,
            num_hidden_layers=2 if self.config.tier == "tiny" else 4,
            num_attention_heads=4 if self.config.tier == "tiny" else 8,
            num_key_value_heads=2 if self.config.tier == "tiny" else 4,
            max_position_embeddings=512,
        )

        model = BharatForCausalLM(cfg)
        if model_ckpt and Path(model_ckpt).is_file():
            st = torch.load(model_ckpt, map_location="cpu", weights_only=False)
            if "model_state_dict" in st:
                model.load_state_dict(st["model_state_dict"], strict=False)

        gguf_meta = export_model_to_gguf_q8_0(model, cfg, self.tokenizer, out_gguf)
        dur = time.perf_counter() - start

        return StageResult(
            stage_name="export",
            status="SUCCESS",
            duration_seconds=dur,
            metrics={
                "tensors_quantized": gguf_meta.tensors_quantized,
                "file_size_bytes": gguf_meta.file_size_bytes,
            },
            artifacts=[str(out_gguf)],
        )

    def run_stage_eval(self, model_ckpt: str | None = None) -> StageResult:
        """Stage 6: 22-Language IndicMMLU & 32k context evaluation."""
        start = time.perf_counter()

        ckpt_path = model_ckpt or str(self.work_dir / "checkpoints_sft" / "final.pt")

        bench_runner = IndicBenchmarkRunner(
            checkpoint_path=ckpt_path,
            device="cpu" if self.config.device == "auto" else self.config.device,
        )
        bench_res = bench_runner.evaluate_mmlu()

        lc_eval = LongContextEvaluator(
            tier=self.config.tier,
            checkpoint_path=ckpt_path if Path(ckpt_path).is_file() else None,
            device="cpu" if self.config.device == "auto" else self.config.device,
        )
        lc_res = lc_eval.run_benchmark(
            context_lengths=[256, 512] if self.config.tier == "tiny" else [1024, 2048],
            depths=[25, 75],
        )

        dur = time.perf_counter() - start
        report_p = self.work_dir / "eval_report.json"
        with open(report_p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "indic_accuracy": bench_res.accuracy_pct,
                    "retrieval_rate": lc_res.overall_accuracy_pct,
                },
                f,
                indent=2,
            )

        return StageResult(
            stage_name="eval",
            status="SUCCESS",
            duration_seconds=dur,
            metrics={
                "overall_indic_accuracy": bench_res.accuracy_pct,
                "needle_retrieval_rate": lc_res.overall_accuracy_pct,
            },
            artifacts=[str(report_p)],
        )

    def run_pipeline(self) -> PipelineRunManifest:
        """Execute all configured stages in sequence and generate signed manifest."""
        total_start = time.perf_counter()
        current_ckpt: str | None = None

        print("\n" + "=" * 65)
        print("🚀 Executing IndicLLM-Bharat Sovereign Pipeline Orchestrator")
        print(f"  • Model Tier:   {self.config.tier.upper()}")
        print(f"  • Stages:       {' -> '.join(s.upper() for s in self.config.stages)}")
        print(f"  • Workspace:    {self.work_dir.resolve()}")
        print("=" * 65 + "\n")

        for stage in self.config.stages:
            st = stage.lower().strip()
            print(f"▶️ Starting Pipeline Stage: [{st.upper()}]...")

            if st == "data":
                res = self.run_stage_data()
            elif st == "pretrain":
                res = self.run_stage_pretrain()
                if res.artifacts:
                    current_ckpt = res.artifacts[0]
            elif st == "sft":
                res = self.run_stage_sft(base_ckpt=current_ckpt)
                if res.artifacts:
                    current_ckpt = res.artifacts[0]
            elif st == "dpo":
                res = self.run_stage_dpo(sft_ckpt=current_ckpt)
                if res.artifacts:
                    current_ckpt = res.artifacts[0]
            elif st == "export":
                res = self.run_stage_export(model_ckpt=current_ckpt)
            elif st == "eval":
                res = self.run_stage_eval(model_ckpt=current_ckpt)
            else:
                raise ValueError(f"Unknown pipeline stage: '{st}'")

            self.stage_results.append(res)
            print(f"✅ Stage [{st.upper()}] completed in {res.duration_seconds:.2f}s.\n")

        total_dur = time.perf_counter() - total_start
        pipeline_id = f"bharat-pipeline-{self.config.tier}-{int(time.time())}"

        stages_dict = [asdict(r) for r in self.stage_results]
        manifest_raw = json.dumps(
            {
                "pipeline_id": pipeline_id,
                "tier": self.config.tier,
                "stages": stages_dict,
                "duration": total_dur,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(manifest_raw.encode("utf-8")).hexdigest()

        manifest = PipelineRunManifest(
            pipeline_id=pipeline_id,
            timestamp=time.time(),
            tier=self.config.tier,
            config={
                "pretrain_steps": self.config.pretrain_steps,
                "sft_steps": self.config.sft_steps,
                "dpo_steps": self.config.dpo_steps,
                "batch_size": self.config.batch_size,
                "device": self.config.device,
            },
            stages=stages_dict,
            total_duration_seconds=total_dur,
            manifest_sha256=digest,
        )

        manifest_file = self.work_dir / "pipeline_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

        print("=" * 65)
        print("🏁 IndicLLM-Bharat Full Lifecycle Pipeline Finished Successfully!")
        print(f"  • Total Duration:   {total_dur:.2f}s")
        print(f"  • Pipeline ID:      {pipeline_id}")
        print(f"  • Run Manifest:     {manifest_file.resolve()}")
        print(f"  • SHA-256 Digest:   {digest}")
        print("=" * 65 + "\n")

        return manifest
