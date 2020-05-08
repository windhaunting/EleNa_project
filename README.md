# EleNa 

We develop a software system for navigating from a source to a destination considering maximum and minimum elevation with a percentage x of shortest path length.

# Architecture:

![Alt text](images/architecture.png?raw=true "architecture")


# How to run:

install Anaconda
install OSMNX follow the instruction https://osmnx.readthedocs.io/en/stable/ 
using following code:
	You can install OSMnx with conda:

	conda config --prepend channels conda-forge
	conda create -n ox --strict-channel-priority osmnx

Other packages needed: tkinter, webbrowser, os, threading, numpy, sys, pickle, heapq, networkx, plotly, folium

How to run:
1. go to Demo directory
2. conda activate ox
3. python EleNa_view.py
