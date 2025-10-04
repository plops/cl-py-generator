# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p02_top.py
import sys
import datetime
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger.info("Logger configured")
# import after logger exists
from p02_impl import GenerationConfig, GenAIJob

# UTC timestamp for output file
timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H_%M_%S")
yaml_filename = f"out_{timestamp}.yaml"
cfg = GenerationConfig(
    prompt_text=r"""Make a summary of recent innovation that came out of Zeiss Meditech""",
    model="gemini-flash-latest",
    output_yaml_path=yaml_filename,
    use_search=True,
    think_budget=-1,
    include_thoughts=True,
)
job = GenAIJob(cfg)
result = job.run()
logger.info(f"thoughts: {result.thoughts}")
logger.info(f"answer: {result.answer}")
logger.info(f"usage: {result.usage_summary}")
