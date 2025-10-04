# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p02_top.py
import sys
import datetime
from loguru import logger
from p02_impl import GenerationConfig, GenAIJob

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger.info("Logger configured")
# UTC timestamp for output file
timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H_%M_%S")
yaml_filename = f"out_{timestamp}.yaml"
cfg = GenerationConfig(
    prompt_text=r"""a discussion about Roger Tsien's predictions 16 years ago lead to this:

#### ** The Funding Driver: Government and Academia as Innovators, Big Pharma as Commercializers:**

Tsien's call for government funding and universities was prescient, as they remain the primary source of truly novel, long-term, and high-risk innovation.

*   **Government/Academic Funding (The """,
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
