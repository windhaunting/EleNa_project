# -*- coding: utf-8 -*-

"""
Created on Tue Mar 24 00:13:25 2020

@author: chao
"""

from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from EleNa_controller import EleNa_Controller
import webbrowser, os
import threading 
import osmnx as ox

api_key = "AIzaSyDVqjj0iKq0eNNHlmslH4fjoFgRj7n-5gs"
controller_obj = EleNa_Controller(api_key)

def reset():
	Start_XY_Entry.delete(0, END)
	Destination_XY_Entry.delete(0, END)
	Tol_Dir.set('')
	Tol_Val.set('')
	best_img.configure(image='')
	adjustment_img.configure(image='')

def helper(source_X, source_Y, sink_X, sink_Y, tol_dir, tol_val):
	controller_obj.cal_route_dij(source_X, source_Y, sink_X, sink_Y, tol_dir, tol_val)

def get_result():
	source_X, source_Y = ox.geo_utils.geocode(Start_XY_Entry.get())
	sink_X, sink_Y = ox.geo_utils.geocode(Destination_XY_Entry.get())
	tol_dir = Tol_Dir.get()
	tol_val = float(Tol_Val.get()[:-1])

	t1 = threading.Thread(target=helper, args=(source_X, source_Y, sink_X, sink_Y, tol_dir, tol_val))
	t1.start()
	t1.join()

	webbrowser.open('file://' + os.path.realpath("data/Best.html"))
	webbrowser.open('file://' + os.path.realpath("data/" + tol_dir + ".html"))

	best = "images/best.png"
	# best = "output/output_1.png"
	best_map = PhotoImage(file=best).subsample(3, 3)
	best_img.configure(image=best_map)
	best_img.image = best_map
	adjustment = "images/" + tol_dir + ".png"
	# adjustment = "output/output_1.png"
	adjustment_map = PhotoImage(file=adjustment).subsample(3, 3)
	adjustment_img.configure(image=adjustment_map)
	adjustment_img.image = adjustment_map

def quit():
	app.destroy()

# Create window object
app = Tk()

Start = Label(app, text='Start Position')
Start.grid(row=0, column=0)
Start_XY = StringVar()
Start_XY_Entry = Entry(app, text=Start_XY)
Start_XY_Entry.grid(row=0, column=1)

Destination = Label(app, text='Destination Position')
Destination.grid(row=1, column=0)
Destination_XY = StringVar()
Destination_XY_Entry = Entry(app, text=Destination_XY)
Destination_XY_Entry.grid(row=1, column=1)

Tolerance = Label(app, text='Tolerance Percentage')
Tolerance.grid(row=2, column=0)
Tol_Dir = ttk.Combobox(app, values=['Min', 'Max'])
Tol_Dir.grid(row=2, column=1)
Tol_Val = ttk.Combobox(app, values=['0%', '25%', '50%'])
Tol_Val.grid(row=2, column=2)

Btn_1 = Button(app, text='    Reset    ', command=reset)
Btn_1.grid(row=3, column=0)
Btn_2 = Button(app, text='    Go!    ', command=get_result)
Btn_2.grid(row=3, column=1)
Btn_3 = Button(app, text='    Quit    ', command=quit)
Btn_3.grid(row=3, column=2)

original = "output/Amherst.png"
original_map = PhotoImage(file=original).subsample(2, 2)
original_img = Label(app, image=original_map)
original_img.grid(row=4, column=0)

best_img = Label(image='')
best_img.grid(row=4, column=1)

adjustment_img = Label(image='')
adjustment_img.grid(row=4, column=2)

app.title('EleNa App')
# app.geometry('700x350')

#Start program
app.mainloop()
