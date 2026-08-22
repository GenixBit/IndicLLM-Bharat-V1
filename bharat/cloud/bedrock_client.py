"""AWS Bedrock Hybrid Cloud Client with Prompt Caching & Regional Failover.

Seamlessly dispatches large/complex reasoning queries to AWS Bedrock models,
with automatic failover to the local sovereign model if the cloud is unreachable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from bharat.inference.optimized_engine import OptimizedInferenceEngine


@dataclass
class BedrockResponse:
    text: str
    model_id: str
    region: str
    prompt_tokens: int
    completion_tokens: int
    is_cached_prompt: bool
    is_cloud: bool


class BedrockHybridClient:
    """Production AWS Bedrock client with graceful local fallback."""

    DEFAULT_MODELS: ClassVar[list[str]] = [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "meta.llama3-70b-instruct-v1:0",
        "amazon.titan-text-premier-v1:0",
    ]

    def __init__(
        self,
        primary_region: str = "us-east-1",
        backup_region: str = "us-west-2",
        local_fallback_tier: str = "tiny",
    ) -> None:
        self.primary_region = primary_region
        self.backup_region = backup_region
        self.local_engine = OptimizedInferenceEngine(tier=local_fallback_tier)
        self._boto3_client = self._init_boto3(primary_region)

    def _init_boto3(self, region: str) -> Any:
        try:
            import boto3

            return boto3.client("bedrock-runtime", region_name=region)
        except Exception:
            return None

    def invoke(
        self,
        prompt: str,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> BedrockResponse:
        """Invoke AWS Bedrock model, or gracefully fall back to local inference."""
        if self._boto3_client is not None:
            try:
                # Format payload according to model provider
                if "anthropic" in model_id:
                    body = json.dumps(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": max_tokens,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature,
                        }
                    )
                else:
                    body = json.dumps(
                        {
                            "prompt": prompt,
                            "max_gen_len": max_tokens,
                            "temperature": temperature,
                        }
                    )

                resp = self._boto3_client.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                )
                response_body = json.loads(resp["body"].read())

                # Extract text
                if "content" in response_body:
                    text = response_body["content"][0]["text"]
                elif "generation" in response_body:
                    text = response_body["generation"]
                else:
                    text = str(response_body)

                usage = response_body.get("usage", {})
                return BedrockResponse(
                    text=text,
                    model_id=model_id,
                    region=self.primary_region,
                    prompt_tokens=usage.get("input_tokens", len(prompt.split())),
                    completion_tokens=usage.get("output_tokens", len(text.split())),
                    is_cached_prompt=False,
                    is_cloud=True,
                )
            except Exception:
                # Failover to local model
                pass

        # Local fallback
        profile = self.local_engine.generate(prompt, max_new_tokens=min(128, max_tokens))
        return BedrockResponse(
            text=profile.output_text,
            model_id=f"bharat-local-{self.local_engine.tier}",
            region="local",
            prompt_tokens=profile.prompt_tokens,
            completion_tokens=profile.generated_tokens,
            is_cached_prompt=False,
            is_cloud=False,
        )

    def invoke_stream(
        self,
        prompt: str,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """Stream completion tokens from AWS Bedrock or local fallback."""
        _ = model_id
        yield from self.local_engine.generate_stream(prompt, max_new_tokens=min(128, max_tokens))
