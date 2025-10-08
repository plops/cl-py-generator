from __future__ import annotations
import os
import time
import asyncio
from dataclasses import dataclass, field, asdict
from typing import List, Any, Dict
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
        safety = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinkingBudget=self.config.think_budget,
                include_thoughts=self.config.include_thoughts,
            ),
            safety_settings=safety,
            tools=tools,
        )
        contents = [
            types.Content(
                role="user", parts=[types.Part.from_text(text=self.config.prompt_text)]
            )
        ]
        logger.debug(f"_build_request {self.config.prompt_text}")
        return dict(
            model=self.config.model, contents=contents, config=generate_content_config
        )

    async def run(self) -> StreamResult:
        req = self._build_request()
        result = StreamResult()
        logger.debug("Starting streaming generation")
        error_in_parts = False
        try:
            async for chunk in self.client.models.generate_content_stream(**req):
                result.responses.append(chunk)
                logger.debug("received chunk")
                try:
                    parts = chunk.candidates[0].content.parts
                except Exception:
                    continue
                try:
                    for part in parts:
                        if getattr(part, "text", None):
                            if getattr(part, "thought", False):
                                result.thought += part.text
                                yield (dict(type="thought", text=part.text))
                            else:
                                result.answer += part.text
                                yield (dict(type="answer", text=part.text))
                except Exception:
                    error_in_parts = True
                    pass
        except Exception as e:
            logger.error(f"genai {e}")
            yield (dict(type="error", message=str(e)))
        logger.debug(f"Thoughts: {result.thoughts}")
        logger.debug(f"Answer: {result.answer}")
        yield (dict(type="complete", thought=result.thought, answer=result.answer))

    def to_dict(self, result: StreamResult) -> Dict[str, Any]:
        return dict(
            config=asdict(self.config), thoughts=result.thoughts, answer=result.answer
        )


__all__ = ["GenerationConfig", "StreamResult", "GenAIJob"]
