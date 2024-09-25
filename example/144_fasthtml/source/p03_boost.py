#!/usr/bin/env python3
from fasthtml.common import *
 
app, rt=fast_app(live=True)
proc_files=["bootconfig", "buddyinfo", "cgroups", "cmdline", "consoles", "cpuinfo", "crypto", "devices", "diskstats", "dma", "execdomains", "filesystems", "interrupts", "iomem", "ioports", "kallsyms", "key-users", "keys", "latency_stats", "loadavg", "locks", "meminfo", "misc", "modules", "mtrr", "pagetypeinfo", "partitions", "schedstat", "slabinfo", "softirqs", "stat", "swaps", "sysrq-trigger", "timer_list", "uptime", "version", "vmstat", "zoneinfo"]
proc_links=[Li(A(f, href=f"/{f}")) for f in proc_files]
nav=Nav(Ul(Li(Strong("Linux Discover"))), Ul(*proc_links, hx_boost=True))
def get_proc_contents(proc):
    lines=""
    if ( ((0)<(len(proc))) ):
        with open(f"/proc/{proc}") as f:
            lines=f.readlines()
    target=f"/{proc}"
    return Pre("".join(lines), hx_get=f"/{proc}/pre", hx_trigger="every 1s", hx_swap="outerHTML")
@rt("/{proc}")
def get(proc: str, request: Request):
    return Titled(f"/{proc}", Main(nav, get_proc_contents(proc), cls="container"))
@rt("/{proc}/pre")
def get(proc: str, request: Request):
    return get_proc_contents(proc)
    lines=""
    if ( ((0)<(len(proc))) ):
        with open(f"/proc/{proc}") as f:
            lines=f.readlines()
    target=f"/{proc}"
    return Pre("".join(lines), hx_get=f"/{proc}/pre", hx_trigger="every 1s", hx_swap="outerHTML")
@rt("/")
def get(request: Request):
    print(request.client.host)
    return Title("Linux Discover Tool"), Main(nav, cls="container")
serve(host="0.0.0.0", port=5001)