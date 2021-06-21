from myhdl import *
from collections import namedtuple

_code_git_version = "81b209f0c0fabd78da477c46760d921fe8377163"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time = "05:09:33 of Monday, 2021-06-21 (GMT+1)"
# https://nbviewer.jupyter.org/github/pcornier/1pCPU/blob/master/pCPU.ipynb
ADM = namedtuple("adm", ["IMP", "IMM", "ABS", "REL", "IDX"])
adm = ADM(*range(5))
OPC = namedtuple("opc", [
    "LDA", "STA", "PHA", "PLA", "ASL", "ASR", "TXA", "TAX", "INX", "DEX",
    "ADD", "SUB", "AND", "OR", "XOR", "CMP", "RTS", "JNZ", "JZ", "JSR", "JMP"
])
opc = OPC(*range(21))


@block
def Gowin_SP_2048x8(clk=None,
                    oce=None,
                    ce=None,
                    reset=None,
                    wre=None,
                    ad=None,
                    din=None,
                    dout=None):
    """similar to https://discourse.myhdl.org/t/instantiating-fpga-components/353/3"""
    @always()
    def NoLogic():
        pass

    sp_inst_0_dout_w = Signal(modbv(0)[23:0])
    sp_inst_0_dout_w.driven = "wire"
    gw_gnd = Signal(bool(0))
    gw_gnd.next = 0
    do_all = concat(sp_inst_0_dout_w, dout)
    blksel = concat(gw_gnd, gw_gnd, gw_gnd)
    ad_all = concat(ad, gw_gnd, gw_gnd, gw_gnd)
    di_all = concat(gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd,
                    gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd,
                    gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd, gw_gnd,
                    gw_gnd, gw_gnd, gw_gnd, din)
    do_all.read = True
    Gowin_SP_2048x8.verilog_code = """SP sp_inst_0(.DO($do_all),.CLK($clk),.OCE($oce),.CE($ce),.RESET($reset),.WRE($wre),.BLKSEL($blksel),.AD($ad_all),.DI($di_all)));"""
    return NoLogic


@block
def mem(clk, adr, we, di, do):
    ram = [Signal(intbv(0)[8:]) for i in range(256)]

    @always(clk.posedge)
    def logic():
        if (we):
            ram[adr.val].next = di
        else:
            if (((adr) < (13))):
                if (((adr.val) == (0))):
                    do.next = 0x1
                elif (((adr.val) == (1))):
                    do.next = 0x0
                elif (((adr.val) == (2))):
                    do.next = 0x38
                elif (((adr.val) == (3))):
                    do.next = 0x0c
                elif (((adr.val) == (4))):
                    do.next = 0x0
                elif (((adr.val) == (5))):
                    do.next = 0x1
                elif (((adr.val) == (6))):
                    do.next = 0x40
                elif (((adr.val) == (7))):
                    do.next = 0x30
                elif (((adr.val) == (8))):
                    do.next = 0x79
                elif (((adr.val) == (9))):
                    do.next = 0x10
                elif (((adr.val) == (10))):
                    do.next = 0x8b
                elif (((adr.val) == (11))):
                    do.next = 0xf8
                elif (((adr.val) == (12))):
                    do.next = 0x0
            else:
                do.next = ram[adr.val]

    return logic


@block
def processor(clk, rst, di, do, adr, we):
    F1, F2, D, E, M1, M2 = range(0, 6)
    pc = Signal(modbv(0)[11:])
    cyc = Signal(modbv(0)[3:])
    ir, im, ra, rx, rw, sr, am, sp = [Signal(modbv(0)[8:]) for i in range(8)]
    sp = Signal(modbv(255)[8:])

    @always(clk.posedge)
    def logic():
        if (rst):
            pc.next = 0
            adr.next = 0
        elif (((cyc) == (F1))):
            adr.next = ((pc) + (1))
            pc.next = ((pc) + (1))
            cyc.next = F2
        elif (((cyc) == (F2))):
            adr.next = ((pc) + (1))
            ir.next = do
            cyc.next = D
        elif (((cyc) == (D))):
            im.next = do
            am.next = ((ir) & (7))
            ir.next = ((((ir) >> (3))) & (31))
            cyc.next = E
        elif (((cyc) == (E))):
            if (((ir) == (opc.LDA))):
                if (((am) == (adm.IMM))):
                    ra.next = im
                    pc.next = ((pc) + (1))
                elif (((am) == (adm.ABS))):
                    adr.next = ((((do) << (8))) | (im))
                    pc.next = ((pc) + (2))
                elif (((am) == (adm.IDX))):
                    adr.next = ((((do) << (8))) | (((im) + (rx))))
                    pc.next = ((pc) + (2))
            elif (((ir) == (opc.STA))):
                if (((am) == (adm.ABS))):
                    adr.next = ((((do) << (8))) | (im))
                    we.next = 1
                    di.next = ra
                    pc.next = ((pc) + (2))
                elif (((am) == (adm.IDX))):
                    adr.next = ((((do) << (8))) | (((im) + (rx))))
                    we.next = 1
                    di.next = ra
                    pc.next = ((pc) + (2))
            elif (((ir) == (opc.JNZ))):
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.TAX))):
                rx.next = ra
                rw.next = 1
            elif (((ir) == (opc.TXA))):
                ra.next = rx
            elif (((ir) == (opc.INX))):
                rx.next = ((rx) + (1))
                rw.next = 1
            elif (((ir) == (opc.CMP))):
                rw.next = 2
                sr.next = concat(((128) <= (((ra) - (im)))),
                                 ((((ra) - (im))) == (0)), sr[6:0])
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.ADD))):
                ra.next = ((ra) + (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.AND))):
                ra.next = ((ra) & (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.OR))):
                ra.next = ((ra) | (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.ASL))):
                ra.next = ((ra) << (im))
                pc.next = ((pc) + (1))
            cyc.next = M1
        elif (((cyc) == (M1))):
            if (True):
                we.next = 0
                adr.next = pc
            cyc.next = M2
        elif (((cyc) == (M2))):
            if (((ir) == (17))):
                ra.next = do
                sr.next = concat(((128) <= (do)), ((do) == (0)), sr[6:0])
            elif (((rw) == (0))):
                sr.next = concat(((128) <= (ra)), ((ra) == (0)), sr[6:0])
            elif (((rw) == (1))):
                sr.next = concat(((128) <= (ra)), ((rx) == (0)), sr[6:0])
            if (((ir) == (23))):
                pc.next = ((((do) << (8))) | (((pc) & (255))))
                adr.next = ((((do) << (8))) | (((pc) & (255))))
            else:
                adr.next = pc
            we.next = 0
            rw.next = 0
            cyc.next = F1

    return logic


def convert_this(hdl):
    clk = Signal(bool(0))
    rst = Signal(bool(1))
    we = Signal(bool(0))
    di = Signal(modbv(0)[8:])
    do = Signal(modbv(0)[8:])
    adr = Signal(intbv(0)[8:])
    ad = Signal(intbv(0)[((1) + (10)):0])
    clk = Signal(bool(0))
    oce = Signal(bool(0))
    ce = Signal(bool(0))
    reset = Signal(bool(0))
    wre = Signal(bool(0))
    mi = mem(clk, adr, we, di, do)
    ram_sp = Gowin_SP_2048x8(clk=clk,
                             oce=oce,
                             ce=ce,
                             reset=rst,
                             wre=wre,
                             ad=ad,
                             din=di,
                             dout=do)
    cpu = processor(clk, rst, di, do, adr, we)
    mi.convert(hdl=hdl)
    cpu.convert(hdl=hdl)
    ram_sp.convert(hdl=hdl)


convert_this(hdl="Verilog")
