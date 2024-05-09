import justpy as jp
_code_git_version="bdcbc4696db16e38cefae324f6b415cd15b111ab"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time="23:38:26 of Thursday, 2024-05-09 (GMT+1)"
start_time=time.time()
debug=True
def hello_world():
    wp=jp.WebPage()
    d=jp.Div(text="hello world")
    wp.add(d)
    return wp
jp.justpy(hello_world)