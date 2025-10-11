from __future__ import annotations
import pandas as pd

# Data Source form here: https://personalitymax.com/personality-types/population-gender/
# They write: To supplement our data, we have also turned to another well-known and authoritative study on gender differences and stereotypes. This normative study was conducted in 1996 by Allen Hammer and Wayne Mitchell, and is titled “The Distribution of Personality Types In General Population.” It surveyed 1267 adults on a number of different demographic factors.
# values are given in percent
df = pd.read_csv("personality_type.csv")
# References (given by gemini 2.5 pro) for matches:
#
# * 1 [dreamsaroundtheworld.com](https://www.dreamsaroundtheworld.com/mbti-compatibility-guide/)
# * 2 [quora.com](https://www.quora.com/What-s-the-best-match-for-an-ESFP)
# * 3 [quora.com](https://www.quora.com/Who-is-the-best-partner-for-an-ENTP)
# * 4 [wikihow.com](https://www.wikihow.com/Entp-Compatibility)
# * 5 [brainmanager.io](https://brainmanager.io/blog/social/entp-compatibility)
# * 6 [cerebralquotient.com](https://www.cerebralquotient.com/blog/mbti/mbti-romantic-compatibility-best-matches-and-apology-styles)
# * 7 [mypersonality.net](https://mypersonality.net/blog/article/entp-compatibility)
# * 8 [personalitymax.com](https://personalitymax.com/personality/esfj/relationships/)
# * 9 [mypersonality.net](https://mypersonality.net/blog/article/isfp-compatibility)
# * 10 [personalitypage.com](https://personalitypage.com/html/ISFP-compatibility.html)
# * 11 [quora.com](https://www.quora.com/What-is-the-best-match-with-INFP)
# * 12 [personalitymax.com](https://personalitymax.com/personality/intp/relationships/)
# * 13 [reddit.com](https://www.reddit.com/r/INTP/comments/th3c5g/what_type_is_the_best_romantic_partner_for_intps/)
# * 14 [personalitypage.com](https://personalitypage.com/html/ESFP-compatibility.html)
# * 15 [reddit.com](https://www.reddit.com/r/ESTJ/comments/7x1lxh/as_an_estj_whos_our_best_and_worst_match/)
# * 16 [personalitypage.com](https://personalitypage.com/html/ESTJ-compatibility.html)
# * 17 [wikihow.com](https://www.wikihow.com/Esfj-Compatibility)
# * 18 [reddit.com](https://www.reddit.com/r/mbti/comments/1ehvlly/best_and_worst_match_for_esfj/)
# * 19 [wikihow.com](https://www.wikihow.com/Infp-Compatibility)
# * 20 [brainmanager.io](https://brainmanager.io/blog/social/infp-compatibility)
# * 21 [quora.com](https://www.quora.com/Who-is-the-best-match-for-an-ENTJ)
#
# It's probably bullshit anyway, but I just want to look at the results
#
dfr = pd.read_csv("romantic_match.csv")
dfn = pd.read_csv("mbti_names.csv")
