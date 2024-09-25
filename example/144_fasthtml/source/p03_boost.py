#!/usr/bin/env python3
# pip install -U google-generativeai python-fasthtml markdown
import re
import markdown
import uvicorn
import sqlite_minutils.db
import datetime
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fasthtml.common import *
 
# open website
# summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table
app, rt=fast_app(live=True)
nav=Nav(Ul(Li(Strong("Linux Discover"))), Ul(Li(A("bootconfig", href="/bootconfig")), Li(A("buddyinfo", href="/buddyinfo")), Li(A("cgroups", href="/cgroups")), Li(A("cmdline", href="/cmdline")), Li(A("consoles", href="/consoles")), Li(A("cpuinfo", href="/cpuinfo")), Li(A("crypto", href="/crypto")), Li(A("devices", href="/devices")), Li(A("diskstats", href="/diskstats")), Li(A("dma", href="/dma")), Li(A("execdomains", href="/execdomains")), Li(A("filesystems", href="/filesystems")), Li(A("interrupts", href="/interrupts")), Li(A("iomem", href="/iomem")), Li(A("ioports", href="/ioports")), Li(A("kallsyms", href="/kallsyms")), Li(A("key-users", href="/key-users")), Li(A("keys", href="/keys")), Li(A("kpagecgroup", href="/kpagecgroup")), Li(A("kpagecount", href="/kpagecount")), Li(A("kpageflags", href="/kpageflags")), Li(A("latency_stats", href="/latency_stats")), Li(A("loadavg", href="/loadavg")), Li(A("locks", href="/locks")), Li(A("meminfo", href="/meminfo")), Li(A("misc", href="/misc")), Li(A("modules", href="/modules")), Li(A("mtrr", href="/mtrr")), Li(A("pagetypeinfo", href="/pagetypeinfo")), Li(A("partitions", href="/partitions")), Li(A("schedstat", href="/schedstat")), Li(A("slabinfo", href="/slabinfo")), Li(A("softirqs", href="/softirqs")), Li(A("stat", href="/stat")), Li(A("swaps", href="/swaps")), Li(A("sysrq-trigger", href="/sysrq-trigger")), Li(A("timer_list", href="/timer_list")), Li(A("uptime", href="/uptime")), Li(A("version", href="/version")), Li(A("vmstat", href="/vmstat")), Li(A("zoneinfo", href="/zoneinfo")), hx_boost=True))
@rt("/{proc}")
def get(proc: str, request: Request):
    print(f"proc={proc} client={request.client.host}")
    lines=""
    if ( ((0)<(len(proc))) ):
        with open(f"/proc/{proc}") as f:
            lines=f.readlines()
    return Titled(f"/{proc}", Main(nav, Pre("".join(lines)), cls="container"))
@rt("/")
def get(request: Request):
    print(request.client.host)
    return Title("Linux Discover Tool"), Main(nav, cls="container")
serve(host="0.0.0.0", port=5001)