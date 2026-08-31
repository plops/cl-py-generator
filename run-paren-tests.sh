#!/bin/sh
# Differential tests for the parenthesis elision of emit-py: every expression is
# emitted with and without :omit-redundant-parentheses and both variants are
# evaluated by python3.  Exits non-zero when the values differ.
sbcl --noinform --disable-debugger \
     --load ~/quicklisp/local-projects/cl-py-generator/paren-tests.lisp \
     --eval '(cl-py-generator/paren-tests::run-paren-tests)' \
     --quit
