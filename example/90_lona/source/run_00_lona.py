import time
from datetime import datetime
from lona.html import HTML, H1, Div
from lona import LonaApp, LonaView
_code_git_version="bdcbc4696db16e38cefae324f6b415cd15b111ab"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time="23:38:39 of Thursday, 2024-05-09 (GMT+1)"
start_time=time.time()
debug=True
app=LonaApp(__file__)
@app.route("/")
class ClockView(LonaView):
    def handle_request(self, request):
        timestamp=Div()
        html=HTML(H1("clock"), timestamp)
        while (True):
            timestamp.set_text(str(datetime.now()))
            self.show(html)
            self.sleep(1)
app.run(port=8080)