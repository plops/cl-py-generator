from myhdl import *
from random import randrange

_code_git_version = "f114546e1715d9e5847695c64e486e6793ebb9ea"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/run_00_flop.py"
_code_generation_time = "08:39:04 of Thursday, 2021-06-10 (GMT+1)"


def dff(q, d, clk):
    @always(clk.posedge)
    def logic():
        q.next = d

    return logic


def test_dff():
    q, d, clk = [Signal(bool(0)) for i in range(3)]
    dff_inst = dff(q, d, clk)

    @always(delay(10))
    def clkgen():
        clk.next = not (clk)

    @always(clk.negedge)
    def stimulus():
        d.next = randrange(2)

    return dff_inst, clkgen, stimulus


def simulate(timesteps):
    tb = traceSignals(test_dff)
    sim = Simulation(tb)
    sim.run(timesteps)


simulate(2000)


def convert():
    q = Signal(bool(0))
    d = Signal(bool(0))
    clk = Signal(bool(0))
    toVerilog(dff, q, d, clk)


convert()
