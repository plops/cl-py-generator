#!/usr/bin/env python3
import webvtt


def parse_vtt_file(filename):
    r"""load vtt from <filename>. Returns deduplicated transcript as string with second-granularity timestamps"""
    old_text = ["__bla__"]
    old_time = "00:00:00"
    out = [dict(text="")]
    # collect transcript. perform deduplication
    for c in webvtt.read(filename):
        if (out[-1]["text"]) != (old_text[-1]):
            out.append(dict(text=old_text[-1], time=old_time))
        old_text = c.text.split("\n")
        old_time = c.start
    ostr = ""
    # skip the first two entries of out (they are left over from the initialization)
    for time_str in out[2:]:
        # cut away the milliseconds from the time stamps
        tstamp_fine = time_str["time"]
        tstamp = tstamp_fine.split(".")[0]
        caption = time_str["text"]
        ostr += f"{tstamp} {caption}\n"
    return ostr
