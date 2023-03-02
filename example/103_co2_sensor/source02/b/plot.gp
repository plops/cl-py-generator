# gnuplot script

set title "Plot of y0, y1, and y2 versus x"
set xlabel "x"
set ylabel "y"
set datafile separator ' ' # set the separator to space
set yrange [1.5:3.5]

do for [i=1:1000] { # repeat 1000 times or until interrupted
    system("./my_exe > gp_data.tmp")
    plot "< tail -n +2 'gp_data.tmp'" using 1:2 with points title "y0" linecolor rgb 'blue', \
         "< tail -n +2 'gp_data.tmp'" using 1:3 with lines title "y1" linecolor rgb 'red', \
         "< tail -n +2 'gp_data.tmp'" using 1:4 with lines title "y2" linecolor rgb 'black'
    pause 1/60.0 # pause for 1/60th of a second
}
