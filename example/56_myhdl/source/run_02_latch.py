from myhdl import *
from random import randrange

_code_git_version = "6314393d5b6c3f618e2e48e67f2eaf23c2dc7d2a"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/run_00_flop.py"
_code_generation_time = "18:58:18 of Thursday, 2021-06-10 (GMT+1)"


# d flip-flop with asynchronous reset
def latch(q, d, g):
    @always_comb
    def logic():
        if (((g) == (1))):
            q.next = d

    return logic


def test_latch():
    q = Signal(bool(0))
    d = Signal(bool(0))
    g = Signal(bool(0))
    inst = latch(q, d, g)

    @always(delay(7))
    def dgen():
        d.next = randrange(2)

    @always(delay(41))
    def ggen():
        g.next = randrange(2)

    return inst, dgen, ggen


def simulate(timesteps):
    tb = traceSignals(test_latch)
    sim = Simulation(tb)
    sim.run(timesteps)


simulate(2000)


def convert():
    q = Signal(bool(0))
    d = Signal(bool(0))
    g = Signal(bool(0))
    toVerilog(latch, q, d, g)


convert()
