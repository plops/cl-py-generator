from myhdl import *
from random import randrange

_code_git_version = "f748fba035146d8126bce58c3f09d079405f9597"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/run_00_flop.py"
_code_generation_time = "23:20:41 of Thursday, 2021-06-10 (GMT+1)"
MAX_COUNT = ((2_000_000) - (1))
counter = Signal(intbv(0, min=0, max=((1) + (MAX_COUNT))))


@block
def leds(sys_clk, sys_rst_n, led):
    @always(sys_clk.posedge, sys_rst_n.negedge)
    def logic():
        if (((sys_rst_n) == (0))):
            counter.next = 0
        else:
            if (((counter) < (MAX_COUNT))):
                counter.next = ((counter) + (1))
            else:
                counter.next = 0

    @always(sys_clk.posedge, sys_rst_n.negedge)
    def logic_led():
        if (((sys_rst_n) == (0))):
            led.next = intbv(6)[3:0]
        else:
            if (((counter) == (MAX_COUNT))):
                led[3:0].next = concat(led[2:0], led[1:0])
            else:
                led.next = led

    return (
        logic,
        logic_led,
    )


def convert_this(hdl):
    sys_clk = Signal(bool(0))
    sys_rst_n = Signal(bool(0))
    led = Signal(intbv(0)[3:])
    leds_1 = leds(sys_clk, sys_rst_n, led)
    leds_1.convert(hdl=hdl)


convert_this(hdl="Verilog")
