from __future__ import annotations
import pandas as pd

# Data Source form here: https://personalitymax.com/personality-types/population-gender/
# They write: To supplement our data, we have also turned to another well-known and authoritative study on gender differences and stereotypes. This normative study was conducted in 1996 by Allen Hammer and Wayne Mitchell, and is titled “The Distribution of Personality Types In General Population.” It surveyed 1267 adults on a number of different demographic factors.
df = pd.reads_csv("personality_type.csv")
