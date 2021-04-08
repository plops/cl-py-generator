import sg
import PySimpleGUI
layout=[[sg.Text("name:")], [sg.Input()], [sg.Button("Ok")]]
window=sg.Window("Window Title", layout)
while (True):
    event, values=window.read()
    if ( (event in (sg.WIN_CLOSED,"Cancel",)) ):
        break
window.close()