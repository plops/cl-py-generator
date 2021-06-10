from myhdl import *
from random import randrange

_code_git_version = "a4df740cf39dab4316176e5d9da77ec9c4b8a30d"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/run_00_flop.py"
_code_generation_time = "18:47:10 of Thursday, 2021-06-10 (GMT+1)"


# d flip-flop with asynchronous reset
def dffa(q, d, clk, rst):
    @always(clk.posedge, rst.negedge)
    def logic():
        if (((rst) == (0))):
            q.next = 0
        else:
            q.next = d

    return logic


def test_dffa():
    q = Signal(bool(0))
    d = Signal(bool(0))
    clk = Signal(bool(0))
    rst = Signal(bool(0))
    dffa_inst = dffa(q, d, clk, rst)

    @always(delay(10))
    def clkgen():
        clk.next = not (clk)

    @always(clk.negedge)
    def stimulus():
        d.next = randrange(2)

    @instance
    def rstgen():
        yield (delay(5))
        rst.next = 1
        while (True):
            yield (delay(randrange(500, 1000)))
            rst.next = 0
            yield (delay(randrange(80, 140)))
            rst.next = 1

    return dffa_inst, clkgen, stimulus, rstgen


def simulate(timesteps):
    tb = traceSignals(test_dffa)
    sim = Simulation(tb)
    sim.run(timesteps)


simulate(2000)


def convert():
    q = Signal(bool(0))
    d = Signal(bool(0))
    clk = Signal(bool(0))
    rst = Signal(bool(0))
    toVerilog(dffa, q, d, clk, rst)


convert()
