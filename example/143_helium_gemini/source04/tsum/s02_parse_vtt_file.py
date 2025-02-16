#!/usr/bin/env python3
import re
def parse_vtt_file(filename):
    r"""load vtt from <filename>. Returns deduplicated transcript as string with second-granularity timestamps"""
    ostr=""
    for c in webvtt.read(sub_file_):
        # we don't need sub-second time resolution. trim it away
        start=c.start.split(".")[0]
        # remove internal newlines within a caption
        cap=c.text.strip().replace("\n", " ")
        # write <start> <c.text> into each line of ostr
        ostr += f"{start} {cap}\n"