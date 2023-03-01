set title "Plot of y0, y1, and y2 versus x"
set xlabel "x"
set ylabel "y"
set datafile separator ' ' # set the separator to space

do for [i=1:1000] { # repeat 1000 times or until interrupted
    set print "<./my_exe" # redirect program output to a temporary file
    plot "< tail -n +2 'gp_data.tmp'" using 1:2 with lines title "y0", \
         "< tail -n +2 'gp_data.tmp'" using 1:3 with lines title "y1", \
         "< tail -n +2 'gp_data.tmp'" using 1:4 with lines title "y2"
    pause 1/60.0 # pause for 1/60th of a second
}