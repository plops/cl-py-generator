from __future__ import annotations
import os
import time
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Callable, Any, Optional, Dict
from loguru import logger
from google import genai
from google.genai import types


@dataclass
class GenerationConfig:
    prompt_text: str
    model: str = "gemini-flash-latest"
    output_yaml_path: str = "out.yaml"
    use_search: bool = True
    think_budget: int = -1
    include_thoughts: bool = True
    api_key_env: str = "GEMINI_API_KEY"


@dataclass
class StreamResult:
    thoughts: str = ""
    answer: str = ""
    responses: List[Any] = field(default_factory=list)
    usage_summary: Dict[str, Any] = field(default_factory=dict)
    first_thought_time: Optional[float] = None
    last_thought_time: Optional[float] = None
    first_answer_time: Optional[float] = None
    final_answer_time: Optional[float] = None
    submit_time: Optional[float] = None

    def timing_metrics(self) -> Dict[str, float]:
        if not (
            (self.first_thought_time)
            and (self.last_thought_time)
            and (self.first_answer_time)
            and (self.final_answer_time)
            and (self.submit_time)
        ):
            return {}
        return dict(
            prompt_parsing_time=((self.first_thought_time) - (self.submit_time)),
            thinking_time=((self.last_thought_time) - (self.first_thought_time)),
            answer_time=((self.final_answer_time) - (self.last_thought_time)),
        )


class UsageAggregator:
    @staticmethod
    def _first(responses, extractor: Callable[[Any], Any]):
        for r in responses:
            try:
                v = extractor(r)
            except Exception:
                v = None
            if v is not None:
                return v
        return None

    @staticmethod
    def _last(responses, extractor: Callable[[Any], Any]):
        for r in reversed(responses):
            try:
                v = extractor(r)
            except Exception:
                v = None
            if v is not None:
                return v
        return None

    @classmethod
    def summarize(cls, result: StreamResult) -> Dict[str, Any]:
        responses = result.responses
        last_with_usage = None
        for r in reversed(responses):
            if getattr(r, "usage_metadata", None) is not None:
                last_with_usage = r
                break
        summary: Dict[str, Any] = {}
        if last_with_usage:
            um = last_with_usage.usage_metadata
            summary["candidates_token_count"] = getattr(
                um, "candidates_token_count", None
            )
            summary["prompt_token_count"] = getattr(um, "prompt_token_count", None)
            summary["thoughts_token_count"] = getattr(um, "thoughts_token_count", None)
        summary["response_id"] = cls._first(
            responses, lambda r: getattr(r, "response_id", None)
        )
        summary["model_version"] = cls._first(
            responses, lambda r: getattr(r, "model_version", None)
        )
        logger.debug(f"model version: {summary['model_version']}")
        totals = [
            getattr(getattr(r, "usage_metadata", None), "total_token_count", None)
            for r in responses
        ]
        numeric_totals = [
            tot
            for tot in totals
            if (
                isinstance(
                    tot,
                    (
                        int,
                        float,
                    ),
                )
            )
        ]
        summary["total_token_count"] = (
            (max(numeric_totals)) if (numeric_totals) else (None)
        )
        summary["finish_reason"] = cls._last(
            responses,
            lambda r: (
                (getattr(r, "finish_reason", None))
                or (
                    (
                        getattr(
                            getattr(r, "candidates", [None])[0], "finish_reason", None
                        )
                    )
                    if (getattr(r, "candidates", None))
                    else (None)
                )
            ),
        )
        # merge timing metrics
        summary.update(result.timing_metrics())
        return summary


class PricingEstimator:
    # Estimates API costs based on token usage and model version. data from https://cloud.google.com/vertex-ai/generative-ai/pricing
    PRICING = {}
    PRICING["gemini-2.5-pro"] = dict(
        input_low=(1.250),
        input_high=(2.50),
        output_low=(10.0),
        output_high=(15.0),
        threshold=200000,
    )
    PRICING["gemini-2.5-flash"] = dict(
        input_low=(0.30),
        input_high=(0.30),
        output_low=(2.50),
        output_high=(2.50),
        threshold=200000,
    )
    PRICING["gemini-2.5-flash-lite"] = dict(
        input_low=(0.10),
        input_high=(0.10),
        output_low=(0.40),
        output_high=(0.40),
        threshold=200000,
    )
    PRICING["gemini-2.0-flash"] = dict(
        input_low=(0.30),
        input_high=(0.30),
        output_low=(2.50),
        output_high=(2.50),
        threshold=200000,
    )
    GROUNDING_PRICING = dict(
        google_search=(35.0),
        web_grounding_enterprise=(45.0),
        google_maps=(25.0),
        grounding_with_data=(2.50),
    )

    @classmethod
    def _normalize_model_name(cls, model_version) -> Optional[str]:
        if not (model_version):
            return None
        m = model_version.lower()
        # check most specific strings first
        if "2.5-pro" in m:
            return "gemini-2.5-pro"
        if "2.5-flash-lite" in m:
            return "gemini-2.5-flash-lite"
        if "2.5-flash" in m:
            return "gemini-2.5-flash"
        if "2.0-flash" in m:
            return "gemini-2.0-flash"
        return None

    @classmethod
    def estimate_cost(
        cls,
        model_version,
        prompt_tokens: Optional[float] = 0,
        thought_tokens: Optional[float] = 0,
        output_tokens: Optional[float] = 0,
        grounding_used=False,
        grounding_type="google_search",
    ) -> Dict[str, Any]:
        model_name = cls._normalize_model_name(model_version)
        if not ((model_name) or (model_name not in cls.PRICING)):
            return dict(
                error=f"Unknown model: {model_version}",
                model_detected=model_name,
                total_cost_usd=(0.0),
            )
        pricing = cls.PRICING[model_name]
        threshold = pricing.get("threshold", float("inf"))
        use_high_tier = (prompt_tokens) > (float(threshold))
        input_rate = (
            (pricing.get("input_high"))
            if (use_high_tier)
            else (pricing.get("input_low"))
        )
        output_rate = (
            (pricing.get("output_high"))
            if (use_high_tier)
            else (pricing.get("output_low"))
        )
        # treat thoughts as part of `output` billing here
        input_cost = ((prompt_tokens) / (1.00e6)) * (input_rate)
        thought_cost = ((thought_tokens) / (1.00e6)) * (output_rate)
        output_cost = ((output_tokens) / (1.00e6)) * (output_rate)
        total_token_cost = (input_cost) + (thought_cost) + (output_cost)
        grounding_cost = 0.0
        grounding_info = {}
        if grounding_used:
            grounding_rate = cls.GROUNDING_PRICING.get(grounding_type, (35.0))
            grounding_cost = (grounding_rate) / (1.00e3)
            grounding_info = dict(
                grounding_type=grounding_type,
                grounding_prompts=1,
                grounding_cost_usd=round(grounding_cost, 6),
                grounding_rate_per_1k=grounding_rate,
                note="Free tier limits apply (not calculated here)",
            )
        total_cost = (total_token_cost) + (grounding_cost)
        result = dict(
            model_version=model_version,
            model_detected=model_name,
            pricing_tier=("high") if (use_high_tier) else ("low"),
            input_tokens=prompt_tokens,
            thought_tokens=thought_tokens,
            output_tokens=output_tokens,
            total_output_tokens=((thought_tokens) + (output_tokens)),
            input_cost_usd=round(input_cost, 6),
            thought_cost_usd=round(thought_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_token_cost_usd=round(total_token_cost, 6),
            total_cost_usd=round(total_cost, 6),
            rates_per_1m=dict(input=input_rate, output=output_rate),
        )
        if grounding_info:
            result["grounding"] = grounding_info
        return result


class GenAIJob:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.client = genai.Client(api_key=os.environ.get(config.api_key_env))

    def _build_request(self) -> Dict[str, Any]:
        tools = (
            ([types.Tool(googleSearch=types.GoogleSearch())])
            if (self.config.use_search)
            else ([])
        )
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinkingBudget=self.config.think_budget,
                include_thoughts=self.config.include_thoughts,
            ),
            tools=tools,
        )
        contents = [
            types.Content(
                role="user", parts=[types.Part.from_text(text=self.config.prompt_text)]
            )
        ]
        return dict(
            model=self.config.model, contents=contents, config=generate_content_config
        )

    def run(self) -> StreamResult:
        req = self._build_request()
        result = StreamResult(submit_time=time.monotonic())
        logger.debug("Starting streaming generation")
        for chunk in self.client.models.generate_content_stream(**req):
            result.responses.append(chunk)
            try:
                parts = chunk.candidates[0].content.parts
            except Exception:
                continue
            try:
                for part in parts:
                    if getattr(part, "text", None):
                        if getattr(part, "thought", False):
                            now = time.monotonic()
                            if result.first_thought_time is None:
                                logger.debug("First thought received")
                                result.first_thought_time = now
                            result.last_thought_time = now
                            result.thoughts += part.text
                        else:
                            now = time.monotonic()
                            if result.first_answer_time is None:
                                logger.debug("First answer chunk received")
                                result.first_answer_time = now
                            result.final_answer_time = now
                            result.answer += part.text
            except Exception:
                pass
        self._persist_yaml(result)
        logger.debug(f"Thoughts: {result.thoughts}")
        logger.debug(f"Answer: {result.answer}")
        result.usage_summary = UsageAggregator.summarize(result)
        u = (result.usage_summary) or ({})
        logger.debug(f"Usage: {result.usage_summary}")
        price = PricingEstimator.estimate_cost(
            model_version=u.get("model_version"),
            prompt_tokens=u.get("prompt_token_count"),
            thought_tokens=u.get("thoughts_token_count"),
            output_tokens=u.get("candidates_token_count"),
            grounding_used=self.config.use_search,
        )
        logger.debug(f"Price: {price}")
        return result

    def _persist_yaml(self, result: StreamResult):
        path = self.config.output_yaml_path
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(result.responses, f, allow_unicode=True, indent=2)
            logger.info(f"Wrote raw responses to {path}")
        except Exception as e:
            logger.error(f"Failed to write YAML: {e}")

    def to_dict(self, result: StreamResult) -> Dict[str, Any]:
        return dict(
            config=asdict(self.config),
            thoughts=result.thoughts,
            answer=result.answer,
            usage=result.usage_summary,
        )


__all__ = [
    "GenerationConfig",
    "StreamResult",
    "GenAIJob",
    "UsageAggregator",
    "PricingEstimator",
]
