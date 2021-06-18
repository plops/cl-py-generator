from myhdl import *
from random import randrange

_code_git_version = "bab0e56393af29adffedb117ec2ecacbbbf66fef"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time = "16:24:21 of Friday, 2021-06-18 (GMT+1)"
# https://tangnano.sipeed.com/en/examples/2_lcd.html
# https://github.com/sipeed/Tang-Nano-examples/blob/master/example_lcd/lcd_pjt/src/VGAMod.v
# AT050TN43.pdf ILI6122.pdf
v_back = 6
v_pulse = 5
v_extent = 272
v_front = 62
h_back = 182
h_pulse = 1
h_extent = 480
h_front = 210
pixel_for_hs = ((h_extent) + (h_back) + (h_front))
line_for_vs = ((v_extent) + (v_back) + (v_front))
pixel_count = Signal(
    intbv(0, min=0, max=((h_extent) + (h_back) + (h_front) + (100))))
line_count = Signal(
    intbv(0, min=0, max=((2) * (((v_extent) + (v_back) + (v_front) + (100))))))
data_r = Signal(intbv(0)[10:])
data_g = Signal(intbv(0)[10:])
data_b = Signal(intbv(0)[10:])


@block
def lcd(pixel_clk, n_rst, lcd_de, lcd_hsync, lcd_vsync, lcd_r, lcd_g, lcd_b):
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

    @always_comb
    def logic_sync():
        if (((((h_pulse) <= (pixel_count))) & (((pixel_count) <=
                                                (((h_extent) + (h_back))))))):
            lcd_hsync.next = 0
        else:
            lcd_hsync.next = 1
        if (((((v_pulse) <= (line_count))) & (((line_count) <=
                                               (line_for_vs))))):
            lcd_vsync.next = 0
        else:
            lcd_vsync.next = 1
        if (((((((h_back) <= (pixel_count))) & (((pixel_count) <=
                                                 (((h_extent) + (h_back)))))))
             & (((((v_back) <= (line_count))) & (((line_count) <=
                                                  (((v_extent) + (5))))))))):
            lcd_de.next = 1
        else:
            lcd_de.next = 0

    @always_comb
    def logic_pattern():
        if (((pixel_count) < (240))):
            lcd_r.next = 1
        elif (((pixel_count) < (480))):
            lcd_g.next = 1
        elif (((pixel_count) < (600))):
            lcd_b.next = 1
        else:
            lcd_r.next = 0
            lcd_g.next = 0
            lcd_b.next = 0

    return (
        logic_count,
        logic_sync,
        logic_pattern,
    )


def convert_this(hdl):
    pixel_clk = Signal(bool(0))
    lcd_de = Signal(bool(0))
    lcd_hsync = Signal(bool(0))
    lcd_vsync = Signal(bool(0))
    lcd_r = Signal(bool(0))
    lcd_g = Signal(bool(0))
    lcd_b = Signal(bool(0))
    n_rst = ResetSignal(0, active=0, isasync=False)
    lcd_1 = lcd(pixel_clk, n_rst, lcd_de, lcd_hsync, lcd_vsync, lcd_r, lcd_g,
                lcd_b)
    lcd_1.convert(hdl=hdl)


convert_this(hdl="Verilog")
