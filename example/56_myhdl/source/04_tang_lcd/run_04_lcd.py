from myhdl import *
from random import randrange

_code_git_version = "502f8925f7c49e318ba29f183e6ac0cbc27fd42d"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/run_00_flop.py"
_code_generation_time = "19:03:45 of Friday, 2021-06-11 (GMT+1)"
# https://tangnano.sipeed.com/en/examples/2_lcd.html
# https://github.com/sipeed/Tang-Nano-examples/blob/master/example_lcd/lcd_pjt/src/VGAMod.v
# AT050TN43.pdf ILI6122.pdf
v_back = intbv(6)
v_pulse = intbv(5)
v_extent = intbv(480)
v_front = intbv(62)
h_back = intbv(182)
h_pulse = intbv(1)
h_extent = intbv(800)
h_front = intbv(210)
pixel_for_hs = ((h_extent) + (h_back) + (h_front))
line_for_vs = ((v_extent) + (v_back) + (v_front))
pixel_count = Signal(
    intbv(0, min=0, max=((h_extent) + (h_back) + (h_front) + (100))))
line_count = Signal(
    intbv(0, min=0, max=((v_extent) + (v_back) + (v_front) + (100))))


@block
def lcd(pixel_clk, n_rst, data_r, data_g, data_b):
    @always(pixel_clk.posedge, n_rst.negedge)
    def logic_count():
        if (((n_rst) == (0))):
            line_count.next = 0
            pixel_count.next = 0
        else:
            if (((pixel_count) == (pixel_for_hs))):
                line_count.next = ((line_count) + (1))
                pixel_count.next = 0
            else:
                if (((line_count) == (line_for_vs))):
                    line_count.next = 0
                    pixel_count.next = 0
                else:
                    pixel_count.next = ((pixel_count) + (1))

    @always(pixel_clk.posedge, n_rst.negedge)
    def logic_data():
        if (((n_rst) == (0))):
            data_r.next = 0
            data_b = 0
            data_g = 0

    return (
        logic_count,
        logic_data,
    )


def convert_this(hdl):
    pixel_clk = Signal(bool(0))
    n_rst = Signal(bool(0))
    data_r = Signal(intbv(0)[5:])
    data_g = Signal(intbv(0)[6:])
    data_b = Signal(intbv(0)[5:])
    lcd_1 = lcd(pixel_clk, n_rst, data_r, data_b, data_g)
    lcd_1.convert(hdl=hdl)


convert_this(hdl="Verilog")
