#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:13:25 2020

@author: fubao
"""

from tkinter import *
from tkinter.filedialog import *
from tkinter.messagebox import *
import tkinter.font as font

from tkinterhtml import HtmlFrame

import urllib.request
from PIL import ImageTk, Image

class EleNaApp(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.master.title("EleNa App")
        self.master.geometry('700x700-0+2')
        
        
        
    def add_input(self):
        
        frmInput = LabelFrame(self.master, text = "Input", padx=10, pady=10)
        #frmInput.pack(side=TOP, expand=Yes, padx=1, pady=2,)
        frmInput.grid(row=0, column=0, columnspan=12, sticky=W)
        
        src = Label(frmInput, text = "Source")
        src.grid(row=0, column=0, rowspan=1, padx=5, pady=5, sticky=W)        
        self.srcEntry = Entry(frmInput, width = 50)
        self.srcEntry.insert(0, " ")  #default 
        self.srcEntry.grid(row=0, column=1, padx=5, pady=5)              

        dst = Label(frmInput, text = "Destination")
        dst.grid(row=1, column=0, rowspan=1, padx=5, pady=5, sticky=W)        
        self.dstEntry = Entry(frmInput, width = 50)
        self.dstEntry.insert(0, " ")  #default 
        self.dstEntry.grid(row=1, column=1, padx=5, pady=5)     
        v = IntVar()

        dst = Label(frmInput, text = "Elevation")
        dst.grid(row=2, column=0, rowspan=1, padx=5, pady=5, sticky=W)    
        values = {"Min " : 1, "Max" : 2}
        for (text, value) in values.items(): 
            elev_btn = Radiobutton(frmInput, text=text, variable=v, value=value)
            elev_btn.grid()   # row=2, column=value, sticky=W)
            #elev_btn  #row=2,  sticky=W  column=value,
        dst = Label(frmInput, text = "Shortest Path X Percentage")
        dst.grid(rowspan=1, padx=5, pady=5, sticky=W)
        dst.place(relx=0.6, rely=0.5, anchor='ne')
        sp_scale = Scale(frmInput, from_=0, to=500, length=200, orient=HORIZONTAL)
        sp_scale.set(100)
        sp_scale.grid(row=3, column=1)
        
        button_text_font = font.Font(size=15)

        submit = Button(frmInput, text ="GO", height = 1, width = 3, bg='#ff0000', fg='#ffffff', command = self.generate_interactive_map)
        submit['font'] = button_text_font

        submit.grid(rowspan=1, padx=5, pady=5, sticky=W)
        submit.place(relx=0.95, rely=0.6, anchor='ne')

        #frame = HtmlFrame(frmInput, horizontal_scrollbar="auto")
        #frame.set_content("<html>aadaaf </html>")
        #frame.set_content(urllib.request.urlopen("http://thonny.cs.ut.ee").read().decode())
        #frame.grid(row=4, column=0, sticky=NSEW)
        frmMap = LabelFrame(self.master, text = "Map", padx=10, pady=10)
        #frmInput.pack(side=TOP, expand=Yes, padx=1, pady=2,)
        frmMap.grid(row=1, column=0, columnspan=12, sticky=W)
        
        path = "Amherst.png"

        load = Image.open(path)
        render = ImageTk.PhotoImage(load.resize((500,400)))
        img = Label(frmMap, image=render)
        img.grid(row=4, column=0,sticky=W)
        img.image = render
        #img.place(x=4, y=0)
        


    def insert_html_map(self):
        # https://stackoverflow.com/questions/46571448/tkinter-and-a-html-file
        frame = HtmlFrame(self.master, horizontal_scrollbar="auto")

        frame.set_content("<html>aadaaf </html>")
        #frame.set_content(urllib.request.urlopen("http://thonny.cs.ut.ee").read().decode())
        print("enter insert_html_map: ")
        
    def generate_interactive_map(self):
        #https://blog.dominodatalab.com/creating-interactive-crime-maps-with-folium/
        x = 1
        
if __name__ == '__main__':

    root = Tk()
    app = EleNaApp(master=root)
    app.add_input()
    app.insert_html_map()
    
    app.mainloop()
    
