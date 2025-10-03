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


class PricingEstimator:
    """Estimates API costs based on token usage and model version."""

    # Pricing per 1M tokens (standard API, non-batch)
    PRICING = {
        "gemini-2.5-pro": {
            "input_low": 1.25,  # <= 200K tokens
            "input_high": 2.5,  # > 200K tokens
            "output_low": 10.0,  # <= 200K input tokens
            "output_high": 15.0,  # > 200K input tokens
            "threshold": 200_000,
        },
        "gemini-2.5-flash": {
            "input_low": 0.30,
            "input_high": 0.30,
            "output_low": 2.50,
            "output_high": 2.50,
            "threshold": 200_000,
        },
        "gemini-2.5-flash-lite": {
            "input_low": 0.10,
            "input_high": 0.10,
            "output_low": 0.40,
            "output_high": 0.40,
            "threshold": 200_000,
        },
        "gemini-2.0-flash": {
            "input_low": 0.30,
            "input_high": 0.30,
            "output_low": 2.50,
            "output_high": 2.50,
            "threshold": 200_000,
        },
    }

    # Grounding costs (per 1,000 grounded prompts, after free tier)
    GROUNDING_PRICING = {
        "google_search": 35.0,  # $35 per 1,000 grounded prompts
        "web_grounding_enterprise": 45.0,  # $45 per 1,000 grounded prompts
        "google_maps": 25.0,  # $25 per 1,000 grounded prompts
        "grounding_with_data": 2.5,  # $2.5 per 1,000 requests
    }

    # Free tier limits per day
    FREE_TIER_LIMITS = {
        "gemini-2.5-pro": {"google_search": 10_000, "google_maps": 10_000},
        "gemini-2.5-flash": {"google_search": 1_500, "google_maps": 1_500},
        "gemini-2.5-flash-lite": {"google_search": 1_500, "google_maps": 1_500},
        "gemini-2.0-flash": {"google_search": 1_500, "google_maps": 1_500},
    }

    @classmethod
    def _normalize_model_name(cls, model_version: str) -> Optional[str]:
        """Extract normalized model name from model_version string."""
        if not model_version:
            return None

        model_lower = model_version.lower()

        # Match model families
        if "2.5-pro" in model_lower or "2.5pro" in model_lower:
            return "gemini-2.5-pro"
        elif "2.5-flash-lite" in model_lower or "2.5flash-lite" in model_lower:
            return "gemini-2.5-flash-lite"
        elif "2.5-flash" in model_lower or "2.5flash" in model_lower:
            return "gemini-2.5-flash"
        elif "2.0-flash" in model_lower or "2.0flash" in model_lower:
            return "gemini-2.0-flash"

        return None

    @classmethod
    def estimate_cost(
        cls,
        model_version: str,
        prompt_tokens: int,
        thought_tokens: int,
        output_tokens: int,
        grounding_used: bool = False,
        grounding_type: str = "google_search",
    ) -> Dict[str, Any]:
        """
        Estimate cost for a generation request.

        Args:
            model_version: The model version string
            prompt_tokens: Number of input/prompt tokens
            thought_tokens: Number of reasoning/thought tokens
            output_tokens: Number of response/output tokens
            grounding_used: Whether grounding was used in this request
            grounding_type: Type of grounding used (google_search, google_maps, etc.)

        Returns:
            Dictionary with cost breakdown and total
        """
        model_name = cls._normalize_model_name(model_version)

        if not model_name or model_name not in cls.PRICING:
            return {
                "error": f"Unknown model: {model_version}",
                "model_detected": model_name,
                "total_cost_usd": 0.0,
            }

        pricing = cls.PRICING[model_name]
        threshold = pricing["threshold"]

        # Determine which tier to use based on input token count
        use_high_tier = prompt_tokens > threshold

        input_rate = pricing["input_high"] if use_high_tier else pricing["input_low"]
        output_rate = pricing["output_high"] if use_high_tier else pricing["output_low"]

        # Calculate token costs (rates are per 1M tokens)
        input_cost = (prompt_tokens / 1_000_000) * input_rate
        thought_cost = (thought_tokens / 1_000_000) * output_rate
        output_cost = (output_tokens / 1_000_000) * output_rate

        total_output_tokens = thought_tokens + output_tokens
        total_token_cost = input_cost + thought_cost + output_cost

        # Calculate grounding cost (if applicable)
        grounding_cost = 0.0
        grounding_info = {}
        if grounding_used:
            # One grounded prompt per request
            grounding_rate = cls.GROUNDING_PRICING.get(grounding_type, 35.0)
            grounding_cost = grounding_rate / 1_000  # Cost per single grounded prompt
            grounding_info = {
                "grounding_type": grounding_type,
                "grounding_prompts": 1,
                "grounding_cost_usd": round(grounding_cost, 6),
                "grounding_rate_per_1k": grounding_rate,
                "note": "Free tier limits apply (not calculated here)",
            }

        total_cost = total_token_cost + grounding_cost

        result = {
            "model_version": model_version,
            "model_detected": model_name,
            "pricing_tier": "high" if use_high_tier else "low",
            "input_tokens": prompt_tokens,
            "thought_tokens": thought_tokens,
            "output_tokens": output_tokens,
            "total_output_tokens": total_output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "thought_cost_usd": round(thought_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_token_cost_usd": round(total_token_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "rates_per_1m": {
                "input": input_rate,
                "output": output_rate,
            },
        }

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
        logger.debug(f"Thoughts: {result.thoughts}")
        logger.debug(f"Answer: {result.answer}")
        result.usage_summary = UsageAggregator.summarize(result)
        logger.debug(f"Usage: {result.usage_summary}")
        self._persist_yaml(result)
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


class UsageAggregator:
    @staticmethod
    def _first(responses, extractor: Callable[[Any], Any]):
        for r in responses:
            try:
                v = extractor(r)
            except Exception:
                v = None
            if not (v is None):
                return v
        return None

    @staticmethod
    def _last(responses, extractor: Callable[[Any], Any]):
        for r in reversed(responses):
            try:
                v = extractor(r)
            except Exception:
                v = None
            if not (v is None):
                return v
        return None

    @classmethod
    def summarize(cls, result: StreamResult) -> Dict[str, Any]:
        responses = result.responses
        last_with_usage = None
        for r in reversed(responses):
            if not (getattr(r, "usage_metadata", None) is None):
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

        # Detect if grounding was used
        grounding_used = False
        grounding_type = "google_search"  # default
        for r in responses:
            candidate = getattr(r, "candidates", [None])[0] if getattr(r, "candidates", None) else None
            if candidate:
                grounding_metadata = getattr(candidate, "grounding_metadata", None)
                if grounding_metadata:
                    # Check various grounding indicators
                    if getattr(grounding_metadata, "web_search_queries", None):
                        grounding_used = True
                        grounding_type = "google_search"
                        break
                    elif getattr(grounding_metadata, "grounding_chunks", None):
                        grounding_used = True
                        grounding_type = "google_search"
                        break

        # Add cost estimation
        if all(
            k in summary
            for k in ["model_version", "prompt_token_count", "thoughts_token_count", "candidates_token_count"]
        ):
            cost_estimate = PricingEstimator.estimate_cost(
                model_version=summary["model_version"],
                prompt_tokens=summary["prompt_token_count"] or 0,
                thought_tokens=summary["thoughts_token_count"] or 0,
                output_tokens=summary["candidates_token_count"] or 0,
                grounding_used=grounding_used,
                grounding_type=grounding_type,
            )
            summary["cost_estimate"] = cost_estimate

        return summary


__all__ = ["GenerationConfig", "StreamResult", "GenAIJob", "UsageAggregator", "PricingEstimator"]
