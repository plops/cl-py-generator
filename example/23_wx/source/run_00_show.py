#!/usr/bin/python3
# pip3 install --user helium
import wx
from helium import *
# %%
_code_git_version="68369002adfc97ff4ffed0c58b9b00fce4ba3dad"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/23_wx/source/run_00_show.py"
_code_generation_time="14:17:34 of Friday, 2020-08-28 (GMT+1)"
# %% https://wxpython.org/pages/overview/index.html 
class HelloFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(HelloFrame, self).__init__(*args, **kw)
        pnl=wx.Panel(self)
        st=wx.StaticText(pnl, label="hello world")
        font=st.GetFont()
        font.PointSize=((font.PointSize)+(10))
        font=font.Bold()
        st.SetFont(font)
        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add(st, wx.SizerFlags().Border(((wx.TOP) | (wx.LEFT)), 25))
        pnl.SetSizer(sizer)
        self.makeMenuBar()
        self.CreateStatusBar()
        self.SetStatusText("Welcome to wxPython")
    def makeMenuBar(self):
        fileMenu=wx.Menu()
        helloItem=fileMenu.Append(-1, "&Hello...\tCtrl-H", "Help string shown in status bar")
        fileMenu.AppendSeparator()
        exitItem=fileMenu.Append(wx.ID_EXIT)
        helpMenu=wx.Menu()
        aboutItem=helpMenu.Append(wx.ID_ABOUT)
        menuBar=wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")
        self.SetMenuBar(menuBar)
        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)
    def OnExit(self, event):
        self.Close(True)
    def OnHello(self, event):
        wx.MessageBox("Hello again")
    def OnAbout(self, event):
        wx.MessageBox("hello sample", "about hello world 2", ((wx.OK) | (wx.ICON_INFORMATION)))
app=wx.App()
frm=HelloFrame(None, title="Hello World 2")
frm.Show()
app.MainLoop()