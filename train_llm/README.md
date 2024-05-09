
# Training settings gemini 1.0

```
Base model: Gemini 1.0 Pro 001 (Tuning)
 Total training time: 12m 51s 
 Tuned examples: 27 examples 
Epochs: 100
 Batch size: 4 
 Learning rate: 0.000080000005 
```

- loss settled after 50 epochs (went from 1400 to 0)


## Try on 24_gtk3

```
Convert the following Python code into s-expressions: 

#!/usr/bin/python3
import wx
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
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
```
Model output:

```lisp
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-py-generator))

(in-package :cl-py-generator)
(progn
  (defvar *path* "/home/martin/stage/cl-py-generator/example/1")
  (defvar *code-file* "gtk3_0_wx_1")
  (defvar *source* (format nil "~a/source/~a" *path* *code-file*))
  (defvar *inspection-facts*
    `((10 "")))

  (let* ((code (with-output-to-string (str)
		  (untranslate-file :source *source*
				    :str    str
				    :encoding 'utf-8
				    :optional ((pyl :skip))))))
	 (eval (format nil "~(~a~%" (join " " (string-split code "\n"))
			 `,(eval code))))))raws
```﻿

- commments:
- this is not good. maybe i trained too long.
- i don't see a way to separate my examples into training data and
  test data when i feed it to the model as csv.
- the structured form has two fields, one for examples and the other for testing
