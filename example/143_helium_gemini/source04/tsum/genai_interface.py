# genai_interface.py
from __future__ import annotations
import os
import types as _types
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Iterator, Union

# Try to import the real google-genai package (optional).
try:
    from google import genai as _genai_pkg  # type: ignore
    from google.genai import types as _genai_types  # type: ignore
    HAS_GENAI = True
except Exception:
    _genai_pkg = None
    _genai_types = None
    HAS_GENAI = False

# Try to detect pydantic BaseModel for better dumping behavior (optional)
try:
    from pydantic import BaseModel as _PydanticBaseModel  # type: ignore
except Exception:
    _PydanticBaseModel = None  # type: ignore

@dataclass
class StreamChunk:
    raw: Any
    text: Optional[str]
    usage: Optional[Dict[str, int]]
    candidates: Optional[List[Any]]

@dataclass
class EmbeddingResult:
    raw: Any
    embedding: List[float]


class GenAIClient:
    """
    Wrapper around google.genai.Client that yields normalized StreamChunk objects
    and returns an EmbeddingResult for embeddings.

    Methods accept either types.GenerateContentConfig objects or plain dicts for config.
    """

    def __init__(self, api_key: Optional[str] = None, debug: bool = False, client: Any = None):
        self.debug = debug
        self._types = _genai_types
        if client is not None:
            self._client = client
        else:
            if not HAS_GENAI:
                raise RuntimeError("google-genai package not available; install 'google-genai' or use MockGenAIClient")
                # Create the real SDK client (it will pick up GEMINI_API_KEY / GOOGLE_API_KEY if api_key is None)
            self._client = _genai_pkg.Client(api_key=api_key) if api_key else _genai_pkg.Client()

            # --------------- Public convenience methods ----------------
    def generate_content_stream(self, model: str, contents: Any, config: Optional[Any] = None) -> Iterator[StreamChunk]:
        """
        Yield StreamChunk for each chunk returned by genai.models.generate_content_stream.
        """
        stream = self._client.models.generate_content_stream(model=model, contents=contents, config=config)
        last = None
        for raw in stream:
            last = raw
            text = self._extract_text(raw)
            usage = self._extract_usage(raw)
            candidates = self._extract_candidates(raw)
            sc = StreamChunk(raw=raw, text=text, usage=usage, candidates=candidates)
            if self.debug:
                print("=== DEBUG: raw chunk introspection ===")
                print(self.pretty_print(raw, max_depth=2))
                print("=== DEBUG: normalized ===")
                print("text:", repr(text))
                print("usage:", usage)
                print("candidates (first 1):", (candidates or [])[:1])
                print("======================================")
            yield sc

    def generate_content(self, model: str, contents: Any, config: Optional[Any] = None) -> Any:
        """
        Non-streaming call; returns the raw response from client.models.generate_content.
        """
        return self._client.models.generate_content(model=model, contents=contents, config=config)

    def embed_content(self, model: str, contents: Any) -> EmbeddingResult:
        """
        Return embedding result as EmbeddingResult(raw=..., embedding=[float,...])
        """
        resp = self._client.models.embed_content(model=model, contents=contents)
        emb = self._extract_embedding(resp)
        if self.debug:
            print("=== DEBUG: embedding response ===")
            print(self.pretty_print(resp, max_depth=2))
            print("extracted vector len:", len(emb) if emb else None)
            print("================================")
        return EmbeddingResult(raw=resp, embedding=emb)

        # --------------- Internal extraction helpers ----------------
    def _pull_text_from_obj(self, obj: Any) -> Optional[str]:
        """
        Heuristic recursive extractor that tries to find text in nested dicts, lists,
        pydantic models, SimpleNamespace-like objects, attributes like .text, .content,
        .parts, or .candidates.
        """
        if obj is None:
            return None
            # primitives
        if isinstance(obj, str):
            return obj
        if isinstance(obj, (int, float, bool)):
            return str(obj)
            # dict
        if isinstance(obj, dict):
            for key in ("text", "content", "message", "delta"):
                if key in obj:
                    v = obj[key]
                    if isinstance(v, str):
                        return v
                    t = self._pull_text_from_obj(v)
                    if t:
                        return t
                        # try parts/candidates/data
            for key in ("parts", "candidates", "messages", "data"):
                v = obj.get(key)
                if isinstance(v, list) and v:
                    return self._pull_text_from_obj(v[0])
                    # fallback: keys with string values
            for k, v in obj.items():
                if isinstance(v, str) and len(v) < 1000:
                    return v
                t = self._pull_text_from_obj(v)
                if t:
                    return t
                    # list/tuple
        if isinstance(obj, (list, tuple)):
            if not obj:
                return None
            return self._pull_text_from_obj(obj[0])
            # pydantic BaseModel
        if _PydanticBaseModel is not None and isinstance(obj, _PydanticBaseModel):
            try:
                dumped = obj.model_dump()
                return self._pull_text_from_obj(dumped)
            except Exception:
                pass
                # objects with attributes
        for attr in ("text", "content", "delta", "message"):
            val = getattr(obj, attr, None)
            if isinstance(val, str):
                return val
            if val is not None:
                t = self._pull_text_from_obj(val)
                if t:
                    return t
                    # parts / candidates attributes
        for attr in ("parts", "candidates", "messages", "data"):
            val = getattr(obj, attr, None)
            if isinstance(val, (list, tuple)) and val:
                return self._pull_text_from_obj(val[0])
                # if object exposes a model_dump or dict
        if hasattr(obj, "model_dump"):
            try:
                return self._pull_text_from_obj(obj.model_dump())
            except Exception:
                pass
        if hasattr(obj, "dict"):
            try:
                return self._pull_text_from_obj(obj.dict())
            except Exception:
                pass
                # fallback to string representation (last resort)
        try:
            s = str(obj)
            if s:
                return s
        except Exception:
            pass
        return None

    def _extract_text(self, chunk: Any) -> Optional[str]:
        return self._pull_text_from_obj(chunk)

    def _extract_usage(self, chunk: Any) -> Optional[Dict[str, int]]:
        """
        Heuristic to extract usage metadata (prompt_token_count, candidates_token_count, thinking_token_count).
        Returns a dict with keys 'prompt','candidates','thinking' (ints) if available.
        """
        candidates = ("usage_metadata", "usage", "response", "response_metadata")
        # check attributes first
        for attr in candidates:
            obj = getattr(chunk, attr, None)
            if obj is None:
                continue
                # dict-like
            if isinstance(obj, dict):
                return {
                    "prompt": int(obj.get("prompt_token_count") or obj.get("promptTokens") or obj.get("prompt") or 0),
                    "candidates": int(obj.get("candidates_token_count") or obj.get("candidateTokens") or obj.get("candidates") or 0),
                    "thinking": int(obj.get("thinking_token_count") or obj.get("thinking") or 0),
                }
                # object-like
            try:
                p = getattr(obj, "prompt_token_count", None) or getattr(obj, "promptTokens", None) or getattr(obj, "prompt", None)
                c = getattr(obj, "candidates_token_count", None) or getattr(obj, "candidateTokens", None) or getattr(obj, "candidates", None)
                t = getattr(obj, "thinking_token_count", None) or getattr(obj, "thinking", None)
                if p is not None or c is not None or t is not None:
                    return {"prompt": int(p or 0), "candidates": int(c or 0), "thinking": int(t or 0)}
            except Exception:
                pass
                # try dict-shaped chunk dumps (pydantic)
        if hasattr(chunk, "model_dump"):
            try:
                d = chunk.model_dump()
                return self._extract_usage(d)
            except Exception:
                pass
        if isinstance(chunk, dict):
            # search deeply for usage keys
            for key in ("usage_metadata", "usage", "response"):
                if key in chunk:
                    return self._extract_usage(chunk[key])
        return None

    def _extract_candidates(self, chunk: Any) -> Optional[List[Any]]:
        """
        Extract candidate objects (raw shapes) if present.
        """
        # dict-like
        if isinstance(chunk, dict):
            c = chunk.get("candidates") or chunk.get("choices") or chunk.get("outputs")
            if isinstance(c, list):
                return c
                # attribute
        c = getattr(chunk, "candidates", None) or getattr(chunk, "choices", None) or getattr(chunk, "outputs", None)
        if isinstance(c, (list, tuple)):
            return list(c)
            # try model_dump
        if hasattr(chunk, "model_dump"):
            try:
                d = chunk.model_dump()
                return self._extract_candidates(d)
            except Exception:
                pass
        return None

    def _extract_embedding(self, resp: Any) -> List[float]:
        """
        Robust extraction of embedding vector from variety of shapes.
        """
        if resp is None:
            return []
            # dict cases
        if isinstance(resp, dict):
            # direct
            if "embedding" in resp and isinstance(resp["embedding"], (list, tuple)):
                return list(resp["embedding"])
                # embeddings: [[...]]
            if "embeddings" in resp and isinstance(resp["embeddings"], list) and resp["embeddings"]:
                first = resp["embeddings"][0]
                if isinstance(first, (list, tuple)):
                    return list(first)
                if isinstance(first, dict) and "embedding" in first:
                    return list(first["embedding"])
                    # data -> first -> embedding
            if "data" in resp and isinstance(resp["data"], list) and resp["data"]:
                first = resp["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return list(first["embedding"])
                    # object-like
        # resp.embedding
        if hasattr(resp, "embedding"):
            val = getattr(resp, "embedding")
            if isinstance(val, (list, tuple)):
                return list(val)
        if hasattr(resp, "embeddings") and getattr(resp, "embeddings"):
            first = getattr(resp, "embeddings")[0]
            if isinstance(first, (list, tuple)):
                return list(first)
            if hasattr(first, "embedding"):
                return list(first.embedding)
        if hasattr(resp, "data") and getattr(resp, "data"):
            first = getattr(resp, "data")[0]
            if isinstance(first, dict) and "embedding" in first:
                return list(first["embedding"])
            if hasattr(first, "embedding"):
                return list(first.embedding)
                # pydantic model_dump fallback
        if hasattr(resp, "model_dump"):
            try:
                return self._extract_embedding(resp.model_dump())
            except Exception:
                pass
        raise RuntimeError("Could not extract embedding vector from embed response; inspect resp.raw using pretty_print")

        # --------------- Debug / pretty-printing ----------------
    def pretty_print(self, obj: Any, max_depth: int = 3, max_items: int = 6) -> str:
        """
        Return a human-friendly representation of the object structure, recursing
        up to max_depth. This is helpful for printing how the real SDK returns objects.
        """
        visited = set()
        out_lines: List[str] = []

        def short_repr(o: Any, max_len: int = 300) -> str:
            try:
                s = repr(o)
            except Exception:
                try:
                    s = str(type(o))
                except Exception:
                    s = "<unreprable>"
            if len(s) > max_len:
                return s[:max_len] + "...(len=%d)" % len(s)
            return s

        def rec(o: Any, depth: int, indent: int):
            prefix = " " * indent
            if id(o) in visited:
                out_lines.append(f"{prefix}<circular id={id(o)} type={type(o).__name__}>")
                return
            visited.add(id(o))
            t = type(o)
            if depth <= 0:
                out_lines.append(f"{prefix}{t.__name__}: {short_repr(o)}")
                return
                # primitives
            if o is None or isinstance(o, (str, int, float, bool)):
                out_lines.append(f"{prefix}{t.__name__}: {short_repr(o)}")
                return
                # dict
            if isinstance(o, dict):
                out_lines.append(f"{prefix}{t.__name__} (len={len(o)}):")
                keys = list(o.keys())[:max_items]
                for k in keys:
                    try:
                        v = o[k]
                    except Exception:
                        v = "<unreadable>"
                    out_lines.append(f"{prefix}  [{k!r}] =>")
                    rec(v, depth - 1, indent + 6)
                if len(o) > max_items:
                    out_lines.append(f"{prefix}  ... {len(o)-max_items} more keys")
                return
                # list/tuple
            if isinstance(o, (list, tuple)):
                out_lines.append(f"{prefix}{t.__name__} (len={len(o)}):")
                for i, v in enumerate(o[:max_items]):
                    out_lines.append(f"{prefix}  [{i}] =>")
                    rec(v, depth - 1, indent + 6)
                if len(o) > max_items:
                    out_lines.append(f"{prefix}  ... {len(o) - max_items} more items")
                return
                # pydantic model
            if _PydanticBaseModel is not None and isinstance(o, _PydanticBaseModel):
                out_lines.append(f"{prefix}{t.__name__} (pydantic model):")
                try:
                    m = o.model_dump()
                    rec(m, depth - 1, indent + 2)
                    return
                except Exception:
                    out_lines.append(f"{prefix}  <failed to model_dump()>")
                    # object with __dict__
            try:
                d = getattr(o, "__dict__", None)
                if isinstance(d, dict) and d:
                    out_lines.append(f"{prefix}{t.__name__} (attrs):")
                    keys = list(d.keys())[:max_items]
                    for k in keys:
                        v = d.get(k)
                        out_lines.append(f"{prefix}  .{k} =>")
                        rec(v, depth - 1, indent + 6)
                    if len(d) > max_items:
                        out_lines.append(f"{prefix}  ... {len(d)-max_items} more attrs")
                    return
            except Exception:
                pass
                # fallback: try attribute names
            try:
                attrs = [a for a in dir(o) if not a.startswith("_")]
                attrs = [a for a in attrs if not callable(getattr(o, a, None))]
                if attrs:
                    out_lines.append(f"{prefix}{t.__name__} (selected attrs):")
                    for a in attrs[:max_items]:
                        try:
                            v = getattr(o, a)
                        except Exception:
                            v = "<unreadable>"
                        out_lines.append(f"{prefix}  .{a} =>")
                        rec(v, depth - 1, indent + 6)
                    if len(attrs) > max_items:
                        out_lines.append(f"{prefix}  ... {len(attrs)-max_items} more attrs")
                    return
            except Exception:
                pass
                # final fallback: short repr
            out_lines.append(f"{prefix}{t.__name__}: {short_repr(o)}")

        rec(obj, max_depth, 0)
        return "\n".join(out_lines)


    # ----------------- Mock client for offline testing -----------------
class MockGenAIClient:
    """
    Produces a set of representative chunk shapes and embedding shapes for offline
    testing / development. Use this when you don't want to call the real API.
    """

    def __init__(self):
        pass

    def generate_content_stream(self, model: str, contents: Any, config: Optional[Any] = None) -> Iterator[Any]:
        """
        Yield a small set of objects that mimic different shapes the real SDK might return.
        These shapes include:
          - SimpleNamespace with text
          - SimpleNamespace with nested .delta or .content.parts[0].text
          - dict-shaped candidate with content.parts[0].text
          - object with .candidates list containing nested objects
        """
        # Simple text chunk (string-like)
        yield SimpleNamespace(text="Mock: part 1 - simpleNamespace.text")
        # Delta-like structure
        yield SimpleNamespace(delta=SimpleNamespace(text="Mock: delta piece"))
        # dict-shaped with nested content.parts
        yield {"candidates": [{"content": {"parts": [{"text": "Mock: dict -> candidates[0].content.parts[0].text"}]}}]}
        # SimpleNamespace containing candidates which contains SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=...)]))
        yield SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="Mock: pydantic-like candidate.parts[0].text")]))])
        # final chunk with usage metadata in different shapes
        yield SimpleNamespace(text="Mock: final chunk", usage_metadata={"prompt_token_count": 10, "candidates_token_count": 50, "thinking_token_count": 2})

    def embed_content(self, model: str, contents: Any) -> Any:
        """
        Return different shaped embeddings depending on the string content passed in, for testing.
        """
        # If the content string contains special markers choose a shape.
        text = contents if isinstance(contents, str) else ""
        if "dict_shape" in text:
            return {"embedding": [0.1, 0.2, 0.3, 0.4]}
        if "embeddings_shape" in text:
            return {"embeddings": [[0.5, 0.6, 0.7]]}
            # SimpleNamespace style
        return SimpleNamespace(embedding=[0.9, 0.8, 0.7, 0.6])