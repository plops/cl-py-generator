# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p02_top.py
import sys
import datetime
from sqlite_minutils import *
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger.info("Logger configured")
from p02_impl import GenerationConfig, GenAIJob

# UTC timestamp for output file
timestamp = datetime.utcnow().strftime("%Y%m%d_%H_%M_%S")
yaml_filename = f"out_{timestamp}.yaml"
cfg = GenerationConfig(
    prompt_text="make a summary of the current state of clinical and experimental cancer treatment. in particular look at the approach roger tien's company uses (fluorescent labels), genetic modification or selection of immune cells, and specialized delivery.",
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
