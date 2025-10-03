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
class StreamResult():
    thoughts-str="q"