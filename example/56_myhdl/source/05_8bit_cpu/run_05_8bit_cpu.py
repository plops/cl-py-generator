from myhdl import *
from collections import namedtuple

_code_git_version = "e2e9b726a5aaf63d94f3f7a08eb99b62f2575bc6"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time = "08:14:17 of Thursday, 2021-06-17 (GMT+1)"
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
    ir, im, rx, rw, sr, am, sp = [Signal(modbv(0)[8:]) for i in range(7)]
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
                addr.next = ((sp) + (1))
                sp.next = ((sp) + (1))
            cyc.next = E

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
