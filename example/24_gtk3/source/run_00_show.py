#!/usr/bin/python3
import wx
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
# %%
_code_git_version="6075d0d96067290f696deb8769c708ce46ec6e9c"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/24_gtk3/source/run_00_show.py"
_code_generation_time="12:01:02 of Friday, 2020-09-11 (GMT+1)"
class ButtonWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="hello world")
        self.button=Gtk.Button(label="click here")
        self.button.connect("clicked", self.on_button_clicked)
        self.add(self.button)
    def on_button_clicked(self, widget):
        print("hello world")
store=Gtk.ListStore(str, str, float)
treeiter=store.append(["art of prog", "knuth", (24.450    )])
tree=Gtk.TreeView(store)