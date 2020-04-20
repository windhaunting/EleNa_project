#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:42:36 2020

@author: fubao
"""
import sys
import osmnx as ox
import networkx as nx
import numpy as np

from plotly import graph_objs as go

import pickle as pkl
from heapq import heappush
from heapq import heappop

class EleNa_Controller(object):
    def __init__(self, key):
        self.key = key
    
    def get_key(self):
        return self.key
    
    
    def get_elevation(self,graph_orig, city_name, key):
        # get evelation data only
        key = self.get_key()
        graph_orig = ox.add_node_elevations(graph_orig, api_key=key)
        graph_orig = ox.add_edge_grades(graph_orig)
        pkl.dump(graph_orig, open("data/" + city_name+"_city_graph.pkl","wb"))

		#projecting map on to 2D space
        graph_project = ox.project_graph(graph_orig)
        pkl.dump(graph_project, open("data/" + city_name+"_city_graph_projected.pkl","wb"))
        return graph_project, graph_orig
    
    def get_map_data_with_elevation(self, city_name, key):
        # get model data from osmnx of 2d and elevation
        
        #city_name = 'Amherst'  # 'Springfield'
        place_query = {'city':city_name, 'state':'Massachusetts', 'country':'USA'}
        graph_orig = ox.graph_from_place(place_query, network_type='walk')
    
        #add Elevation data from GoogleMaps
        key = self.get_key()
        graph_orig = ox.add_node_elevations(graph_orig, api_key=key)  # 'AIzaSyDVqjj0iKq0eNNHlmslH4fjoFgRj7n-5gs')   # from elevation
        graph_orig = ox.add_edge_grades(graph_orig)
        pkl.dump(graph_orig, open("data/" + city_name+"_city_graph.pkl","wb"))
    
        #project map on to 2D space
        graph_project = ox.project_graph(graph_orig)
        pkl.dump(graph_project, open("data/" + city_name+"_city_graph_projected.pkl","wb"))
        #print ("pkl: ", type(graph_orig))
        return graph_project, graph_orig
    
    def get_dist_cost(self, graph_project, node_a, node_b):
        return graph_project.edges[node_a, node_b, 0]['length']
    
    def get_elevation_cost(self, graph_project, node_a, node_b):
        # get each 
        return (graph_project.nodes[node_a]['elevation'] - graph_project.nodes[node_b]['elevation'])
    
    
    def get_total_elevation(self, graph_projection, route):
        # get total evelvation
        if not route:
            return 0
        elevation_cost = 0
        for i in range(len(route)-1):
            elevation_data = self.get_elevation_cost(graph_projection, route[i], route[i+1])
            if elevation_data > 0:
                elevation_cost += elevation_data
        return elevation_cost
    
    
    def ground_truth_shorest_route(self, graph, source, destination, weight = 'length'):
        # route of ground truth shortest path
        route = nx.shortest_path(graph, source, destination, weight = 'length')
        print ("ground_truth_shorest_route: ", route)
        return route
    
    def get_ground_truth_shorest_length(self, graph, source, destination, weight = 'length'):
        # get length of shortest path
        return nx.shortest_path_length(graph, source=source, target=destination, weight=weight, method='dijkstra')


    # algorithm: dijkstra
    def get_shortest_dijkstra_route(self, graph, source=0, destination=0, weight='length'):
        #algorithm 1:  consider length or elevation as weight cost
        
        frontier = []     # min heap
        heappush(frontier, (0, source))
        came_from = {}
        cost_so_far = {}
        came_from[source] = None
        cost_so_far[source] = 0
		
        while len(frontier) != 0:
            (dist, current) = heappop(frontier)
            if current == destination:
                break;
            for u, next, data in graph.edges(current, data=True):
                new_cost = cost_so_far[current]
                if weight == 'length':
                    inc_cost = self.get_dist_cost(graph, u, next)
                elif weight == 'elevation':
                    inc_cost = self.get_elevation_cost(graph, u, next)
                if inc_cost > 0:
                    new_cost += inc_cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost
                    heappush(frontier, (priority, next))
                    came_from[next] = current
                    
        route_by_length_min_elevaltion = []
        p = destination
        route_by_length_min_elevaltion.append(p)
        while p != source:
            p = came_from[p]
            route_by_length_min_elevaltion.append(p)
        
        route_by_length_min_elevaltion = route_by_length_min_elevaltion[::-1]
            
        print ("get_shortest_dijkstra_route: ", route_by_length_min_elevaltion)
        return route_by_length_min_elevaltion
    
 
    def get_dijkstra_evelation_shorest_perentage_route(self, graph, source, destination, allowed_cost, elevation_mode='minimize'):
        # control percentage
        frontier = []
        heappush(frontier, (0, source))
        came_from = {}
        cost_so_far = {}
        cost_so_far_ele = {}
        came_from[source] = None
        cost_so_far[source] = 0
        cost_so_far_ele[source] = 0
        while len(frontier) != 0:
            (dist, current) = heappop(frontier)
            if current == destination:
                if cost_so_far[current] <= allowed_cost:
                    break
            for u, next, data in graph.edges(current, data=True):
                new_cost = cost_so_far[current] + self.get_dist_cost(graph, current, next)
                new_cost_ele = cost_so_far_ele[current]
                elevation_cost = self.get_elevation_cost(graph, current, next)
                if elevation_cost > 0:
                    new_cost_ele = new_cost_ele + elevation_cost 
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far_ele[next] = new_cost_ele
                    cost_so_far[next] = new_cost
                    priority = new_cost_ele
                    if elevation_mode =='maximize':
                        priority = priority
                    heappush(frontier, (priority, next))
                    came_from[next] = current
                    
        route_by_length_minele = []
        p = destination
        route_by_length_minele.append(p)
        while p != source:
            p = came_from[p]
            route_by_length_minele.append(p)
        route_by_length_minele = route_by_length_minele[::-1]
        print ("get_dijkstra_evelation_shorest_perentage_route: ", route_by_length_minele)
        return route_by_length_minele

    def read_map_data(self, download_flag, graph_origin_file, graph_project_file):
        # if already downloaded
        if download_flag:  # read from graph file
            graph_project, graph_orig = self.get_map_data_with_elevation()
        
        else:
            with open(graph_project_file, 'rb') as graph_project_obj:
    
                #graph_orig = pkl.load(pkl_graph)
                graph_project = pkl.load(graph_project_obj)
    
                #print ("type: ", type(graph_project), graph_project.graph)
            with open(graph_origin_file, 'rb') as graph_origin_obj:
    
                #graph_orig = pkl.load(pkl_graph)
                graph_orig = pkl.load(graph_origin_obj)            
        return graph_project, graph_orig
    
    
    
    def plot_path(self, lat_list, long_list, src_lat_long, destination_lat_long):
        #Given a list of latitudes and longitudes, origin and destination latitude and longitude, plots a path on a map   

        # adding the lines joining the nodes
        fig = go.Figure(go.Scattermapbox(
            name = "Path",
            mode = "lines",
            lon = long_list,
            lat = lat_list,
            marker = {'size': 10},
            line = dict(width = 4.5, color = 'blue')))
        # adding source marker
        fig.add_trace(go.Scattermapbox(
            name = "Source",
            mode = "markers",
            lon = [src_lat_long[1]],
            lat = [src_lat_long[0]],
            marker = {'size': 12, 'color':"red"}))
         
        # adding destination marker
        fig.add_trace(go.Scattermapbox(
            name = "Destination",
            mode = "markers",
            lon = [destination_lat_long[1]],
            lat = [destination_lat_long[0]],
            marker = {'size': 12, 'color':'green'}))
        
        # getting center for plots:
        lat_center = np.mean(lat_list)
        long_center = np.mean(long_list)
        # defining the layout using mapbox_style
        fig.update_layout(mapbox_style="stamen-terrain",
            mapbox_center_lat = 30, mapbox_center_lon=-80)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                          mapbox = {
                              'center': {'lat': lat_center, 
                              'lon': long_center},
                              'zoom': 13})
        fig.show()
    
    
    def plot_one_route(self, graph, route, src_lat_long, destination_lat_long):
        
        long = [] 
        lat = []  
        for i in route:
             point = graph.nodes[i]
             long.append(point['x'])
             lat.append(point['y'])
     
        #self.plot_path(lat, long, src_lat_long, destination_lat_long)

    
        # create route colors
        rc1 = ['r'] * (len(route) - 1)
        rc2 = ['b'] * len(route)
        rc = rc1 + rc2
        nc = ['r', 'r', 'b', 'b']
        
    
    def plot_two_routes(self, graph, route1, route2, src_lat_long, destination_lat_long):
        
        
        #self.plot_path(lat, long, src_lat_long, destination_lat_long)

        
        # create route colors
        rc1 = ['r'] * (len(route1) - 1)
        rc2 = ['b'] * len(route2)
        rc = rc1 + rc2
        nc = ['r', 'r', 'b', 'b']
        
        # plot the routes
        fig, ax = ox.plot_graph_routes(graph, [route1, route2], route_color=rc, orig_dest_node_color=nc, node_size=0)
        


    def test_dijkstra(self):
        
        src_lat_long = (42.406670, -72.531005)
        destination_lat_long = (42.325745, -72.531929) # (42.376796, -72.501432)
         
        graph_origin_file = "data/Amherst_city_graph.pkl"
        graph_project_file = "dataAmherst_city_graph_projected.pkl"  # "Amherst_city_graph_projected.pkl"
        graph_project, graph_orig = controller_obj.read_map_data(False, graph_origin_file, graph_project_file)
        source = ox.get_nearest_node(graph_orig, (src_lat_long))
        destination = ox.get_nearest_node(graph_orig, (destination_lat_long))
        
        print ("graph_project.source: ", source)
        print ("graph_project.dst: ", destination)
        route1 = self.ground_truth_shorest_route(graph_orig, source=source, destination=destination, weight='length')
        
        #route2 = self.get_shortest_dijkstra_route(graph_orig, source=source, destination=destination, weight='length')
    
        shortest_path_length = self.get_ground_truth_shorest_length(graph_orig, source, destination)       # self.get_total_length(graph_projection, shortest_path)
        overhead = 50
        allowed_cost = ((100.0 + overhead)*shortest_path_length)/100.0
        
        elevation_mode = "maximize"
        route2 = self.get_dijkstra_evelation_shorest_perentage_route(graph_orig, source, destination, allowed_cost, elevation_mode=elevation_mode)
        
        self.plot_two_routes(graph_orig, route1, route2, src_lat_long, destination_lat_long)

        
    def test2(self):
        
        src_lat_long = (42.406670, -72.531005)
        destination_lat_long = (42.325745, -72.531929)
   
        graph_origin_file = "data/Amherst_city_graph.pkl"
        graph_project_file = "data/Amherst_city_graph_projected.pkl"  # "Amherst_city_graph_projected.pkl"
        graph_project, graph_orig = controller_obj.read_map_data(False, graph_origin_file, graph_project_file)
        source = ox.get_nearest_node(graph_orig, (src_lat_long))
        destination = ox.get_nearest_node(graph_orig, (destination_lat_long))
        
        print ("graph_project.source: ", source)
        print ("graph_project.dst: ", destination)
        
        #print ("nodes: ", graph_orig.nodes)
        print("nodes: ", len(graph_orig.nodes(data=True)), list(graph_orig.nodes(data=True))[1], list(graph_orig.edges(data=True))[1])      # nodes:  5333

        print(list(graph_orig.edges(data=True))[1][2]['geometry'])
    
        ox.plot_graph(graph_orig, node_color='green')
        
        #route1 = self.ground_truth_shorest_route(graph_orig, source=source, destination=destination, weight='length')
        #route2 = self.get_shortest_dijkstra_route(graph_orig, source=source, destination=destination, weight='elevation')
        #self.plot_route(graph_orig, route, src_lat_long, destination_lat_long)
        
        #self.plot_two_routes(graph_orig, route2, route2, src_lat_long, destination_lat_long)
        
        
    
if __name__== "__main__": 
    api_key = "AIzaSyDVqjj0iKq0eNNHlmslH4fjoFgRj7n-5gs"
    controller_obj = EleNa_Controller(api_key)
    

    # download map data
    """
    city_name = 'Amherst'
    graph_project, graph_orig = controller_obj.get_map_data_with_elevation(city_name, api_key)
    
    print ("graph_project.nodes: ", graph_project.nodes)
    
    graph_project_file = "Northampton_city_graph_projected.pkl"  # "Amherst_city_graph_projected.pkl"
    graph_project = controller_obj.read_map_data(False, graph_project_file)
    node_a = 68845560
    node_b = 1709506533
    controller_obj.get_elevation_cost(graph_project, node_a, node_b)
    """
    
    #controller_obj.test_dijkstra()
    controller_obj.test2()