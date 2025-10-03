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
    responses: list[Any] = field(default_factory=list)
    usage_summary: Dict[str, Any] = field(default_factory=dict)
    first_thought_time: Optional[float] = None
    last_thought_time: Optional[float] = None
    first_answer_time: Optional[float] = None
    final_answer_time: Optional[float] = None
    submit_time: Optional[float] = None
