# The --eval runs the test function, and --quit exits SBCL afterwards.
# The script will exit with a non-zero status if tests fail.
sbcl --load ~/quicklisp/local-projects/cl-py-generator/transpiler-tests.lisp \
     --eval '(cl-py-generator/tests::run-transpiler-tests)' \
     --quit
