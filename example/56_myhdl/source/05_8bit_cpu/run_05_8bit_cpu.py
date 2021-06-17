from myhdl import *
from collections import namedtuple

_code_git_version = "50f5d9a2d312db491314871f0618937266fc8f2f"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time = "09:07:42 of Thursday, 2021-06-17 (GMT+1)"
# https://nbviewer.jupyter.org/github/pcornier/1pCPU/blob/master/pCPU.ipynb
ADM = namedtuple("adm", ["IMP", "IMM", "ABS", "REL", "IDX"])
adm = ADM(*range(5))
OPC = namedtuple("opc", [
    "LDA", "STA", "PHA", "PLA", "ASL", "ASR", "TXA", "TAX", "INX", "DEX",
    "ADD", "SUB", "AND", "OR", "XOR", "CMP", "RTS", "JNX", "JZ", "JSR", "JMP"
])
opc = OPC(*range(21))
rom = (
    0x1,
    0x0,
    0x38,
    0x0c,
    0x0,
    0x1,
    0x40,
    0x30,
    0x79,
    0x10,
    0x8b,
    0xf8,
    0x0,
)


@block
def mem(clk, adr, we, di, do):
    ram = [Signal(intbv(0)[8:]) for i in range(256)]

    @always(clk.posedge)
    def logic():
        if (we):
            ram[adr.val].next = di
        else:
            if (((adr) < (len(rom)))):
                do.next = 0
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
            if (((((ir) >> (3))) == (opc.RTS))):
                adr.next = ((sp) + (1))
                sp.next = ((sp) + (1))
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
            elif (((ir) == (opc.TAX))):
                rx.next = ra
                rw.next = 1
            elif (((ir) == (opc.TXA))):
                ra.next = rx
            elif (((ir) == (opc.INX))):
                rx.next = ((rx) + (1))
                rw.next = 1
            elif (((ir) == (opc.DEX))):
                rx.next = ((rx) - (1))
                rw.next = 1
            elif (((ir) == (opc.PHA))):
                adr.next = sp
                sp.next = ((sp) - (1))
                di.next = ra
                we.next = 1
            elif (((ir) == (opc.PLA))):
                sp.next = ((sp) + (1))
                adr.next = ((sp) + (1))
            elif (((ir) == (opc.CMP))):
                rw.next = 2
                sr.next = concat(((128) <= (((ra) - (im)))),
                                 ((((ra) - (im))) == (0)), sr[6:0])
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.JSR))):
                adr.next = sp
                sp.next = ((sp) - (1))
                di.next = ((((pc) + (2))) >> (8))
                we.next = 1
            elif (((ir) == (opc.RTS))):
                adr.next = ((sp) + (1))
                sp.next = ((sp) + (1))
            elif (((ir) == (opc.JMP))):
                pc.next = ((((do) << (8))) | (im))
            elif (((ir) == (opc.ADD))):
                ra.next = ((ra) + (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.SUB))):
                ra.next = ((ra) - (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.AND))):
                ra.next = ((ra) & (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.OR))):
                ra.next = ((ra) | (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.XOR))):
                ra.next = ((ra) ^ (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.ASL))):
                ra.next = ((ra) << (im))
                pc.next = ((pc) + (1))
            elif (((ir) == (opc.ASR))):
                ra.next = ((ra) >> (im))
                pc.next = ((pc) + (1))
            cyc.next = M1

    return logic


def convert_this(hdl):
    clk = Signal(bool(0))
    rst = Signal(bool(1))
    we = Signal(bool(0))
    adr = Signal(modbv(0)[16:])
    di = Signal(modbv(0)[8:])
    do = Signal(modbv(0)[8:])
    mi = mem(clk, adr, we, di, do)
    cpu = processor(clk, rst, di, do, adr, we)
    mi.convert(hdl=hdl)
    cpu.convert(hdl=hdl)


convert_this(hdl="Verilog")
