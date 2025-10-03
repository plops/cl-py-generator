import os
import time
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Callable, Any, Optional, Dict
from __future__ import annotations
from loguru import logger
from google import genai
from google.genai import types
@dataclass
class GenerationConfig():
    prompt_text:str
    model:str="gemini-flash-latest"
    output_yaml_path:str="out.yaml"
    use_search:bool=True
    think_budget:int=-1
    include_thoughts:bool=True
    api_key_env:str="GEMINI_API_KEY"
@dataclass
class StreamResult():
    thoughts:str=""
    answer:str=""
    responses:list[Any]=field(default_factory=list)
    usage_summary:Dict[str,Any]=field(default_factory=dict)
    first_thought_time:Optional[float]=None
    last_thought_time:Optional[float]=None
    first_answer_time:Optional[float]=None
    final_answer_time:Optional[float]=None
    submit_time:Optional[float]=None
    def timing_metrics(self)->Dict[str,float]:
        if ( not(((self.first_thought_time) and (self.last_thought_time) and (self.first_answer_time) and (self.final_answer_time) and (self.submit_time))) ):
            return {}
        return dict(prompt_parsing_time=((self.first_thought_time)-(self.submit_time)), thinking_time=((self.last_thought_time)-(self.first_thought_time)), answer_time=((final_answer_time)-(last_thought_timeq)))
class UsageAggregator():
    @staticmethod
    def _first(responses, extractor: Callable[[Any],Any]):
        for r in responses:
            try:
                v=extractor(r)
            except Exception:
                v=None
            if ( not((v is None)) ):
                return v
        return None
    @staticmethod
    def _last(responses, extractor: Callable[[Any],Any]):
        for r in reversed(responses):
            try:
                v=extractor(r)
            except Exception:
                v=None
            if ( not((v is None)) ):
                return v
        return None
    @classmethod
    def summarize(cls, result: StreamResult)->Dict[str,Any]:
        responses=result.responses
        last_with_usage=None
        for r in reversed(responses):
            if ( not((getattr(r, "usage_metadata", None) is None)) ):
                last_with_usage=r
                break
        summary: Dict[str,Any]={}
        if ( last_with_usage ):
            um=last_with_usage.usage_metadata
            summary["candidates_token_count"]=getattr(um, "candidates_token_count", None)
            summary["prompt_token_count"]=getattr(um, "prompt_token_count", None)
            summary["thoughts_token_count"]=getattr(um, "thoughts_token_count", None)
        summary["response_id"]=cls._first(responses, lambda r: getattr(r, "response_id", None))
        summary["model_version"]=cls._first(responses, lambda r: getattr(r, "model_version", None))
        numeric_totals=[(tot) if (isinstance(tot, (int,float,))) for tot in totals]
        summary["total_token_count"]=(max(numeric_totals)) if (numeric_totals) else (None)
        finish_reason[]=cls._last(responses, lambda r: ((getattr(r, "finish_reason", None)) or ((getattr(getattr(r, "candidates", [None])[0], "finish_reason", None)) if (getattr(r, "candidates", None)) else (None))))
        # merge timing metrics
        summary.update(result.timing_metrics())
        return summary