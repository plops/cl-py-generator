
import yaml
from unittest.mock import MagicMock
from typing import List, Any, Generator

class MockModels:
    """Mocks the 'models' attribute of the genai Client."""

    def __init__(self, responses: List[Any]):
        self._responses = responses

    def generate_content_stream(self, **kwargs) -> Generator[Any, None, None]:
        """
        A generator that yields responses loaded from the YAML file,
        simulating the streaming behavior of the real API.
        """
        yield from self._responses

class MockClient:
    """Mocks the genai.Client."""

    def __init__(self, api_key: str = None, responses: List[Any] = None):
        # The api_key is ignored, but included for signature matching.
        self.models = MockModels(responses)

class MockGenAI:
    """
    A mock implementation of the 'genai' module.

    It's designed to be initialized with data from a YAML file and then
    used to patch the real 'genai' module in tests.
    """
    def __init__(self, yaml_playback_file: str):
        """
        Initializes the mock by loading response data from a YAML file.

        Args:
            yaml_playback_file: Path to the YAML file containing the list of
                                serialized GenerateContentResponse objects.
        """
        with open(yaml_playback_file, 'r', encoding='utf-8') as f:
            # UnsafeLoader is required to deserialize Python objects from YAML.
            self._responses = yaml.unsafe_load(f)

        # This makes MockGenAI.Client(...) return our mocked client.
        self.Client = lambda api_key: MockClient(api_key=api_key, responses=self._responses)

        # To ensure that `from google import genai` and subsequent calls to
        # `genai.types` work, we need to attach the real 'types' module
        # to our mock. This avoids having to mock every single type used.
        try:
            from google.genai import types
            self.types = types
        except ImportError:
            # If the google.genai library isn't installed, create a mock 'types'
            self.types = MagicMock()
