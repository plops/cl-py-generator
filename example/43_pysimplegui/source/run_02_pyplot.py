import PySimpleGUIWeb as sg
import numpy as np
import matplotlib.backends.backend_tkagg
import matplotlib.figure
import matplotlib
import matplotlib.pyplot as plt
import io
layout=[[sg.Image(key="-IMAGE-")], [sg.Button("Draw"), sg.Button("Exit")]]
window=sg.Window("plot example", layout)
while (True):
    event, values=window.read()
    if ( (event in (sg.WIN_CLOSED,"Exit",)) ):
        break
    if ( ((event)==("Draw")) ):
        plt.close("all")
        fig=plt.figure(figsize=[5, 4], dpi=72)
        x=np.linspace(0, 3, 100)
        fig.add_subplot(111).plot(x, np.sin(((2)*(np.pi)*(x))))
        canv=matplotlib.backends.backend_tkagg.FigureCanvasAgg(plt.gcf())
        buf=io.BytesIO()
        canv.print_figure(buf, format="png")
        if ( (buf is None) ):
            print(problem)
        buf.seek(0)
        window["-IMAGE-"].update(data=buf.read())
window.close()