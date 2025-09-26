# test_genai_stream_debug.py
import os
from genai_interface import GenAIClient, MockGenAIClient

def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    force_mock = os.getenv("FORCE_MOCK", "0") == "1"
    use_mock = force_mock or (api_key is None)

    model = os.getenv("TEST_GENAI_MODEL", "gemini-2.5-flash-preview-09-2025")
    prompt = "Write a short story about a robot and a cat in 80 words."

    if use_mock:
        print("Using MockGenAIClient (no network).")
        client = MockGenAIClient()
    else:
        print("Using real GenAIClient (will call the API).")
        client = GenAIClient(api_key=api_key, debug=True)

    print("Streaming generation (showing introspection of raw chunks)...\n")
    try:
        for i, chunk in enumerate(client.generate_content_stream(model=model, contents=prompt, config=None)):
            print("------ CHUNK #%d ------" % i)
            print("Normalized text ->", repr(chunk.text))
            print("Normalized usage ->", chunk.usage)
            print("Raw object type:", type(chunk.raw))
            print("Raw object introspection:\n")
            # pretty_print is available on GenAIClient; Mock client doesn't provide pretty_print,
            # so call client's pretty_print if available, else show repr.
            pp = getattr(client, "pretty_print", None)
            if pp:
                print(pp(chunk.raw, max_depth=2))
            else:
                print(repr(chunk.raw))
            if i >= 10:
                print("Stopped after 10 chunks.")
                break
    except Exception as e:
        print("Error during streaming test:", e)


if __name__ == "__main__":
    main()