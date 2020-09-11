#!/usr/bin/python3
import wx
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
# %%
_code_git_version="f0da680e7d9ccda832f1eeb9cbde63c7967fd76a"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/24_gtk3/source/run_00_show.py"
_code_generation_time="11:51:15 of Friday, 2020-09-11 (GMT+1)"
class ButtonWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="hello world")
        self.button=Gtk.Button(label="click here")
        self.button.connect("clicked", self.on_button_clicked)
        self.add(self.button)
    def on_button_clicked(self, widget):
        print("hello world")
win=ButtonWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()