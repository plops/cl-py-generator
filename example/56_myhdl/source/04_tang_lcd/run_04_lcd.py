from myhdl import *
from random import randrange

_code_git_version = "758f23daea1fcd938219354b807ffceb851be680"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/run_00_flop.py"
_code_generation_time = "19:51:23 of Friday, 2021-06-11 (GMT+1)"
# https://tangnano.sipeed.com/en/examples/2_lcd.html
# https://github.com/sipeed/Tang-Nano-examples/blob/master/example_lcd/lcd_pjt/src/VGAMod.v
# AT050TN43.pdf ILI6122.pdf
v_back = 6
v_pulse = 5
v_extent = 480
v_front = 62
h_back = 182
h_pulse = 1
h_extent = 800
h_front = 210
pixel_for_hs = ((h_extent) + (h_back) + (h_front))
line_for_vs = ((v_extent) + (v_back) + (v_front))
pixel_count = Signal(
    intbv(0, min=0, max=((h_extent) + (h_back) + (h_front) + (100))))
line_count = Signal(
    intbv(0, min=0, max=((v_extent) + (v_back) + (v_front) + (100))))
data_r = Signal(intbv(0)[10:])
data_g = Signal(intbv(0)[10:])
data_b = Signal(intbv(0)[10:])


@block
def lcd(pixel_clk, n_rst, lcd_de, lcd_hsync, lcd_vsync, data_r, data_g,
        data_b):
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
            data_b.next = 0
            data_g.next = 0

    @always_comb
    def logic_sync():
        if (((((h_pulse) <= (pixel_count))) & (((pixel_count) <=
                                                (((h_extent) + (h_back))))))):
            lcd_hsync = 0
        else:
            lcd_hsync = 1
        if (((((v_pulse) <= (line_count))) & (((line_count) <=
                                               (line_for_vs))))):
            lcd_vsync = 0
        else:
            lcd_vsync = 1
        if (((((((h_back) <= (pixel_count))) & (((pixel_count) <=
                                                 (((h_extent) + (h_back)))))))
             & (((((v_back) <= (line_count))) & (((line_count) <=
                                                  (((v_extent) + (5))))))))):
            lcd_de = 1
        else:
            lcd_de = 0

    @always_comb
    def logic_pattern():
        if (((pixel_count) < (200))):
            lcd_r.next = 0
        else:
            if (((pixel_count) < (240))):
                lcd_r.next = 1
            else:
                if (((pixel_count) < (280))):
                    lcd_r.next = 2
                else:
                    if (((pixel_count) < (320))):
                        lcd_r.next = 4
                    else:
                        if (((pixel_count) < (360))):
                            lcd_r.next = 8
                        else:
                            if (((pixel_count) < (400))):
                                lcd_r.next = 16
                            else:
                                lcd_r.next = 0
        if (((pixel_count) < (440))):
            lcd_b.next = 0
        else:
            if (((pixel_count) < (480))):
                lcd_b.next = 1
            else:
                if (((pixel_count) < (520))):
                    lcd_b.next = 2
                else:
                    if (((pixel_count) < (560))):
                        lcd_b.next = 4
                    else:
                        if (((pixel_count) < (600))):
                            lcd_b.next = 8
                        else:
                            if (((pixel_count) < (640))):
                                lcd_b.next = 16
                            else:
                                lcd_b.next = 0
        if (((pixel_count) < (640))):
            lcd_g.next = 0
        else:
            if (((pixel_count) < (680))):
                lcd_g.next = 1
            else:
                if (((pixel_count) < (720))):
                    lcd_g.next = 2
                else:
                    if (((pixel_count) < (760))):
                        lcd_g.next = 4
                    else:
                        if (((pixel_count) < (800))):
                            lcd_g.next = 8
                        else:
                            if (((pixel_count) < (840))):
                                lcd_g.next = 16
                            else:
                                lcd_g.next = 0

    return (
        logic_count,
        logic_data,
        logic_sync,
        logic_pattern,
    )


def convert_this(hdl):
    pixel_clk = Signal(bool(0))
    lcd_de = Signal(bool(0))
    lcd_hsync = Signal(bool(0))
    lcd_vsync = Signal(bool(0))
    n_rst = ResetSignal(0, active=0, isasync=False)
    lcd_r = Signal(intbv(0)[5:])
    lcd_g = Signal(intbv(0)[6:])
    lcd_b = Signal(intbv(0)[5:])
    lcd_1 = lcd(pixel_clk, n_rst, lcd_de, lcd_hsync, lcd_vsync, lcd_r, lcd_b,
                lcd_g)
    lcd_1.convert(hdl=hdl)


convert_this(hdl="Verilog")
