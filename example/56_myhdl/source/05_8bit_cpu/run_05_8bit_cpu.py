from myhdl import *
from collections import namedtuple

_code_git_version = "c86f4edfee3c7e448bb955cb833bedf86df930d8"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time = "13:00:31 of Saturday, 2021-06-19 (GMT+1)"
# https://nbviewer.jupyter.org/github/pcornier/1pCPU/blob/master/pCPU.ipynb
ADM = namedtuple("adm", ["IMP", "IMM", "ABS", "REL", "IDX"])
adm = ADM(*range(5))
OPC = namedtuple("opc", [
    "LDA", "STA", "PHA", "PLA", "ASL", "ASR", "TXA", "TAX", "INX", "DEX",
    "ADD", "SUB", "AND", "OR", "XOR", "CMP", "RTS", "JNZ", "JZ", "JSR", "JMP"
])
opc = OPC(*range(21))
rom_data = (
    1,
    0,
    56,
    12,
    0,
    1,
    64,
    48,
    121,
    16,
    139,
    248,
    0,
)


@block
def rom(clk, adr, do, CONTENT):
    @always_comb
    def read():
        do.next = CONTENT[int(adr)]

    return read


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
    rom_1 = rom(clk, adr, do, rom_data)
    mi = mem(clk, adr, we, di, do)
    cpu = processor(clk, rst, di, do, adr, we)
    rom_1.convert(hdl=hdl)


convert_this(hdl="Verilog")
