#docker run -it \
#  -e GEMINI_API_KEY="your_actual_gemini_api_key" \
#  -v /path/to/your/local/code:/workspace/src \
#  antigravity-sandbox:26.04


# Create the folder on your host first so Docker doesn't generate it as 'root'
mkdir -p "$HOME/.gemini"

docker run -it \
  -e ANTIGRAVITY_PLAINTEXT_AUTH=1 \
  -v "$HOME/.gemini:/root/.gemini" \
  -v "/home/kiel/stage:/workspace/src" \
  antigravity-sandbox:26.04