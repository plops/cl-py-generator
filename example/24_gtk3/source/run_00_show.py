#!/usr/bin/python3
import wx
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
# %%
_code_git_version="e57a1a0f6fb8b29675f9338afef98a60cd98a0a2"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/24_gtk3/source/run_00_show.py"
_code_generation_time="12:24:46 of Friday, 2020-09-11 (GMT+1)"
class TreeViewWindow(Gtk.Window):
    def language_filter_func(self, model, iter, data):
        if ( ((self.current_filter_language is None) or (((self.current_filter_language)==("None")))) ):
            return True
        else:
            return ((model[iter][2])==(self.current_filter_language))
    def __init__(self):
        Gtk.Window.__init__(self, title="hello world")
        self.store=Gtk.ListStore(str, int, str)
        for ref in [("firefox",2002,"c++",), ("emacs",1984,"lisp",)]:
            self.store.append(list(ref))
        self.filter=self.store.filter_new()
        self.current_filter_language=None
        self.filter.set_visible_func(self.language_filter_func)
        self.treeview=Gtk.TreeView.new_with_model(self.filter)
        for i, column_title in enumerate(["software", "release_year", "language"]):
            renderer=Gtk.CellRendererText()
            column=Gtk.TreeViewColumn(column_title, renderer, text=i)
            self.treeview.append_column(column)
        self.scroll=Gtk.ScrolledWindow()
        self.scroll.set_vexpand(True)
        self.add(self.scroll)
        self.scroll.add(self.treeview)
        self.show_all()
win=TreeViewWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()