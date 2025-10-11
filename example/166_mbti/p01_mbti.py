from __future__ import annotations
import pandas as pd

# Data Source form here: https://personalitymax.com/personality-types/population-gender/
# They write: To supplement our data, we have also turned to another well-known and authoritative study on gender differences and stereotypes. This normative study was conducted in 1996 by Allen Hammer and Wayne Mitchell, and is titled “The Distribution of Personality Types In General Population.” It surveyed 1267 adults on a number of different demographic factors.
# values are given in percent
df = pd.read_csv("personality_type.csv")
#
# >>> df
#     TYPE  TOTAL  MALE  FEMALE
# 0   INTJ    2.1   3.3     0.9
# 1   INTP    3.3   4.8     1.7
# 2   ENTJ    1.8   2.7     0.9
# 3   ENTP    3.2   4.0     2.4
# 4   INFJ    1.5   1.2     1.6
# 5   INFP    4.4   4.1     4.6
# 6   ENFJ    2.5   1.6     3.3
# 7   ENFP    8.1   6.4     9.7
# 8   ISTJ   11.6  16.4     6.9
# 9   ISFJ   13.8   8.1    19.4
# 10  ESTJ    8.7  11.2     6.3
# 11  ESFJ   12.0   7.5    16.9
# 12  ISTP    5.4   8.5     2.3
# 13  ISFP    8.8   7.6     9.9
# 14  ESTP    4.3   5.6     3.0
# 15  ESFP    8.5   6.9    10.1
#
#
df = df.set_index("TYPE")
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
#
# >>> dfr
#    mbti_type best_romantic_matches  references
# 0       ISTJ            ESFP, ESTP         NaN
# 1       ISFJ            ESFP, ESTP         1,2
# 2       INFJ            ENFP, ENTP   1,3,4,5,6
# 3       INTJ            ENFP, ENTP     1,4,5,7
# 4       ISTP            ESFJ, ESTJ           1
# 5       ISFP      ENFJ, ESFJ, ESTJ    1,8,9,10
# 6       INFP            ENFJ, ENTJ        1,11
# 7       INTP            ENTJ, ESTJ     1,12,13
# 8       ESTP            ISFJ, ISTJ         NaN
# 9       ESFP            ISFJ, ISTJ        1,14
# 10      ENFP            INFJ, INTJ           1
# 11      ENTP            INFJ, INTJ     1,3,4,7
# 12      ESTJ      ISTP, ISFP, INTP     1,15,16
# 13      ESFJ            ISTP, ISFP   1,8,17,18
# 14      ENFJ            INFP, ISFP     1,19,20
# 15      ENTJ            INFP, INTP  1,12,13,21
#
dfr["best_romantic_matches"] = dfr["best_romantic_matches"].str.split(", ")
# I want a table with how easy it is for a person to find a romantic match, e.g. male ENTP -> female INFJ or female INTJ -> 1.6 + 0.9 = 2.5%
dfr["male_finds_female_match_%"] = dfr["best_romantic_matches"].apply(
    lambda matches: df.loc[matches, "FEMALE"].sum()
)
dfn = pd.read_csv("mbti_names.csv")
#    mbti_type        name
# 0       ISTJ   Inspector
# 1       ISFJ   Protector
# 2       INFJ    Advocate
# 3       INTJ   Architect
# 4       ISTP     Crafter
# 5       ISFP      Artist
# 6       INFP    Mediator
# 7       INTP     Thinker
# 8       ESTP      Dynamo
# 9       ESFP   Performer
# 10      ENFP    Champion
# 11      ENTP     Debater
# 12      ESTJ  Supervisor
# 13      ESFJ    Provider
# 14      ENFJ       Giver
# 15      ENTJ   Commander
#
# for readabilty add the name to dfr
dfr = pd.merge(dfr, dfn, on="mbti_type")
dfr["female_finds_male_match_%"] = dfr["best_romantic_matches"].apply(
    lambda matches: df.loc[matches, "MALE"].sum()
)
dfr["male_has_it_easier"] = (
    (dfr["female_finds_male_match_%"]) < (dfr["male_finds_female_match_%"])
)
result_df_male = dfr[
    ["mbti_type", "name", "male_finds_female_match_%", "male_has_it_easier"]
]
result_df_female = dfr[
    ["mbti_type", "name", "female_finds_male_match_%", "male_has_it_easier"]
]
print(result_df_male.sort_values(by="male_finds_female_match_%"))
print(result_df_female.sort_values(by="female_finds_male_match_%"))
#
#    mbti_type  male_finds_female_match_%
# 10      ENFP                        2.5
# 11      ENTP                        2.5
# 6       INFP                        4.2
# 15      ENTJ                        6.3
# 7       INTP                        7.2
# 3       INTJ                       12.1
# 2       INFJ                       12.1
# 13      ESFJ                       12.2
# 1       ISFJ                       13.1
# 0       ISTJ                       13.1
# 12      ESTJ                       13.9
# 14      ENFJ                       14.5
# 4       ISTP                       23.2
# 8       ESTP                       26.3
# 9       ESFP                       26.3
# 5       ISFP                       26.5
#    mbti_type  female_finds_male_match_%
# 6       INFP                        4.3
# 11      ENTP                        4.5
# 10      ENFP                        4.5
# 15      ENTJ                        8.9
# 2       INFJ                       10.4
# 3       INTJ                       10.4
# 14      ENFJ                       11.7
# 1       ISFJ                       12.5
# 0       ISTJ                       12.5
# 7       INTP                       13.9
# 13      ESFJ                       16.1
# 4       ISTP                       18.7
# 5       ISFP                       20.3
# 12      ESTJ                       20.9
# 8       ESTP                       24.5
# 9       ESFP                       24.5
#
#
