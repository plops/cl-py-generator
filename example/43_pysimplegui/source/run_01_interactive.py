import PySimpleGUIWeb as sg
layout=[[sg.Text("name:")], [sg.Input(key="-INPUT-")], [sg.Text(size=(40,1,), key="-OUTPUT-")], [sg.Button("Ok"), sg.Button("Quit")]]
window=sg.Window("Window Title", layout)
while (True):
    event, values=window.read()
    if ( (event in (sg.WIN_CLOSED,"Quit",)) ):
        break
    window["-OUTPUT-"].update((("Hello ")+(values["-INPUT-"])+("! Thanks for trying.")))
window.close()