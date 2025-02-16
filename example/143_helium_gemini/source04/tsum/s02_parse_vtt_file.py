#!/usr/bin/env python3
import webvtt
def parse_vtt_file(filename):
    r"""load vtt from <filename>. Returns deduplicated transcript as string with second-granularity timestamps"""
    ostr=""
    old_text=""
    for c in webvtt.read(filename):
        start=c.start
        duplicated_p=c.text.strip().startswith(old_text.strip())
        if ( duplicated_p ):
            # the line containing a duplication will be stored in cap. remove internal newlines within a caption.
            cap=c.text.strip().replace("\n", " ")
            print("nil duplicated_p={} start={} cap={}".format(duplicated_p, start, cap))
            # write <start> <c.text> into each line of ostr
            ostr += f"{start} {cap}\n"
        else:
            cap=""
        old_text=c.text
    return ostr