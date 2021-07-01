from nmigen import *

_code_git_version = "422749e6d3534e96bb6625b1cef28b339c014701"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/58_migen/source/00_first/run_00.py"
_code_generation_time = "03:04:04 of Thursday, 2021-07-01 (GMT+1)"


class UpCounter(Elaboratable):
    def __init__(self, limit):
        self.limit = limit
        self.en = Signal()
        self.ovf = Signal()
        self.count = Signal(16)

    def elaborate(self, platform):
        m = Module()
        m.d.comb = ((m.d.comb) + (self.ovf.eq(((self.count) == (self.limit)))))
        with m.If(self.en):
            with m.If(self.ovf):
                incf(m.d.sync, self.count.eq(0))
