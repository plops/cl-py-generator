#!/usr/bin/python3
import wx
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
# %%
_code_git_version="637eafbead5ed8bf2eb2fbf71d4e2588b84950a8"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/24_gtk3/source/run_00_show.py"
_code_generation_time="11:49:10 of Friday, 2020-09-11 (GMT+1)"
win=Gtk.Window()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()