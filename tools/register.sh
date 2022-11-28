# on android with termux and ecl quicklisp doesn't seem can't register directories in ~/quicklisp/local-projects (as of 2022-06-20)

# as a work around i copy this file into
# ~/quicklisp/local-projects
# and execute
find */*.asd > system-index.txt
