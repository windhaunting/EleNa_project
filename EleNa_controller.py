#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:42:36 2020

@author: fubao
"""
import sys
import folium
import osmnx as ox
import networkx as nx
import numpy as np
import math

from collections import defaultdict
from plotly import graph_objs as go

import pickle as pkl
from heapq import heappush
from heapq import heappop


class AStarNode(object):
    # for A star cost, distance to source, heuristic to h
    def __init__(self):
        self.f = 0
        self.g = 0
        self.h = 0
        
        
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



    #BFS
    def exmaple(self, graph, source, target, allowed_cost):
        target = 6604284938
        unvisited = [(source, 0, allowed_cost, [source])]
        visited = []
        route_list = []
        while len(unvisited) > 0:

            #print("loop")
            cur, ele_cost, dis_cost_left, route = unvisited[0]
            unvisited = unvisited[1:]
            #print(route)
            if cur == target:
                route_list.append(route)
                break
            else:
                if dis_cost_left >= 0:
                    for u, next, data in graph.edges(cur, data=True):
                        if next not in route:
                            new_ele_cost = self.get_elevation_cost(graph, u, next)
                            distance_cost = self.get_dist_cost(graph, u, next)
                            unvisited.append((next, ele_cost+new_ele_cost, dis_cost_left - distance_cost, route +[next]))

        # print(route_list[0])
        # print(len(route_list))
        # def myFuc(e):
        #     return e[1]
        #
        # route_list = sorted(route_list, key=myFuc, reverse=True)
        # return route_list[0][3]
        print(len(route_list))
        return(route_list[0])

    def test_BFS(self):

        src_lat_long = (42.406670, -72.531005)
        destination_lat_long = (42.325745, -72.531929)  # (42.376796, -72.501432)

        graph_origin_file = "data/Amherst_city_graph.pkl"
        graph_project_file = "data/Amherst_city_graph_projected.pkl"  # "Amherst_city_graph_projected.pkl"
        graph_project, graph_orig = controller_obj.read_map_data(False, graph_origin_file, graph_project_file)
        source = ox.get_nearest_node(graph_orig, (src_lat_long))
        destination = ox.get_nearest_node(graph_orig, (destination_lat_long))

        print("graph_project.source: ", source)
        print("graph_project.dst: ", destination)
        route1 = self.ground_truth_shorest_route(graph_orig, source=source, destination=destination, weight='length')

        # route2 = self.get_shortest_dijkstra_route(graph_orig, source=source, destination=destination, weight='length')

        shortest_path_length = self.get_ground_truth_shorest_length(graph_orig, source,
                                                                    destination)  # self.get_total_length(graph_projection, shortest_path)
        overhead = 50
        allowed_cost = ((100.0 + overhead) * shortest_path_length) / 100.0

        elevation_mode = "maximize"
        route2 = self.exmaple(graph_orig, source, destination, allowed_cost)


        # route4 = self.get_a_star_shorest_perentage_route(graph_orig, source, destination, allowed_cost, heuristic=heuristic, elevation_mode=elevation_mode)

        self.plot_two_routes(graph_orig, route1, route2, src_lat_long, destination_lat_long)
    # algorithm: dijkstra


     # algorithm: dijkstra, not controlling of shortest path percentage
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
                    
        route_by_length_min_elev = []
        p = destination
        route_by_length_min_elev.append(p)
        while p != source:
            p = came_from[p]
            route_by_length_min_elev.append(p)
        
        route_by_length_min_elev = route_by_length_min_elev[::-1]
            
        print ("get_shortest_dijkstra_route: ", route_by_length_min_elev)
        return route_by_length_min_elev
    

    #algorithm: dijkstra,  control percentage of shortest path
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
            for u, next_node, data in graph.edges(current, data=True):
                new_cost = cost_so_far[current] + self.get_dist_cost(graph, current, next_node)
                new_cost_ele = cost_so_far_ele[current]
                elevation_cost = self.get_elevation_cost(graph, current, next_node)
                new_cost_ele_min = new_cost_ele 
                new_cost_ele_max = new_cost_ele
                if elevation_mode =='minimize':
                    if elevation_cost > 0:
                        new_cost_ele_min = new_cost_ele 
                    else:
                        new_cost_ele_min = new_cost_ele + elevation_cost
                        
                if elevation_mode =='maximize':
                    if elevation_cost > 0:
                        new_cost_ele_max = new_cost_ele + elevation_cost 
                    else:
                        new_cost_ele_max = new_cost_ele
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far_ele[next_node] = new_cost_ele
                    cost_so_far[next_node] = new_cost
                    priority = new_cost_ele_min
                    if elevation_mode =='maximize':
                        priority = new_cost_ele_max
                    heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
                    
        route_by_length_min_elev = []
        p = destination
        route_by_length_min_elev.append(p)
        while p != source:
            p = came_from[p]
            route_by_length_min_elev.append(p)
        route_by_length_min_elev = route_by_length_min_elev[::-1]
        print ("get_dijkstra_evelation_shorest_perentage_route: ", route_by_length_min_elev)
        return route_by_length_min_elev

    def straight_line_length(self, graph, src_node, dst_node):
        # straight line distance around the earth  as the heuristic
        
        lat1 = graph.nodes[src_node]['y']  # lat_long_src[0]
        lon1 = graph.nodes[src_node]['x']
        
        lat2 = graph.nodes[dst_node]['y']
        lon2 = graph.nodes[dst_node]['x']
        
        p = math.pi/180
        a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) * math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2
        R = 6371  # km
        return 2*R * math.asin(math.sqrt(a)) #2*R*asin...


    # algorithm: A*, control percentage of shortest path
    def get_A_star_evelation_shorest_perentage_route(self, graph, source, destination, allowed_cost, heuristic='shortest_path', elevation_mode='minimize'):
        frontier = []
        heappush(frontier, (0, source))
        came_from = {}
        cost_so_far = defaultdict(AStarNode)  # {}         f, g, h  ; f = g + h
        cost_so_far_ele = defaultdict(AStarNode) # {}
        came_from[source] = None
        cost_so_far[source] = AStarNode()
        cost_so_far_ele[source] = AStarNode()
        while len(frontier) != 0:
            (dist, current) = heappop(frontier)
            if current == destination:
                #print (" get_A_star_evelation_shorest_perentage_route new_cost_f: ", current)
                if cost_so_far[current].g <= allowed_cost:
                    break
            #print (" get_A_star_evelation_shorest_perentage_route new_cost_f: ", len(frontier))

            for u, next_node, data in graph.edges(current, data=True):
                new_cost_g = cost_so_far[current].g + self.get_dist_cost(graph, current, next_node)
                #route_g = self.ground_truth_shorest_route(graph, next, destination, weight = None)
                if heuristic == 'straight-line':   # straight-line distance to the goal
                    length_g = self.straight_line_length(graph, next_node, destination)
                elif heuristic == 'shortest-path':   # shortest path to the goal
                    length_g = self.get_ground_truth_shorest_length(graph, next_node, destination, weight = None)
                new_cost_f = new_cost_g + length_g 
                
                elevation_cost_g = self.get_elevation_cost(graph, current, next_node)
                new_cost_ele_g = cost_so_far_ele[current].g
                
                new_cost_ele_f = new_cost_ele_g  #  + elevation_cost_g
                new_cost_ele_min_g = new_cost_ele_g
                new_cost_ele_max_g = new_cost_ele_g
                
                if elevation_mode == 'minimize':
                    if elevation_cost_g > 0:
                        new_cost_ele_min_g = new_cost_ele_g
                    else:
                        new_cost_ele_min_g = new_cost_ele_g + elevation_cost_g
                
                if elevation_mode == 'maximize':
                    if elevation_cost_g > 0:
                        new_cost_ele_max_g = new_cost_ele_g + elevation_cost_g 
                    else:
                        new_cost_ele_max_g = new_cost_ele_g
                
                if next_node not in cost_so_far or new_cost_f < cost_so_far[next_node].f:
                    cost_so_far_ele[next_node].g = new_cost_ele_g
                    cost_so_far_ele[next_node].f = new_cost_ele_f
                    cost_so_far[next_node].g = new_cost_g
                    cost_so_far[next_node].f = new_cost_f
                    #print (" get_A_star_evelation_shorest_perentage_route next: ", len(frontier), next)
                    priority = new_cost_ele_min_g
                    if elevation_mode =='maximize':
                        priority = new_cost_ele_max_g
                    heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
                    
        route_by_length_min_elev = []
        p = destination
        route_by_length_min_elev.append(p)
        while p != source:
            p = came_from[p]
            route_by_length_min_elev.append(p)
        route_by_length_min_elev = route_by_length_min_elev[::-1]
        print ("get_A_star_evelation_shorest_perentage_route: ", route_by_length_min_elev)
        return route_by_length_min_elev

        
        
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
        graph_project_file = "data/Amherst_city_graph_projected.pkl"  # "Amherst_city_graph_projected.pkl"
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
        elevation_mode = "minimize"
        route3 = self.get_dijkstra_evelation_shorest_perentage_route(graph_orig, source, destination, allowed_cost, elevation_mode=elevation_mode)



    def test_Atar(self):
        
        src_lat_long = (42.406670, -72.531005)
        destination_lat_long = (42.325745, -72.531929) # (42.376796, -72.501432)
         
        graph_origin_file = "data/Amherst_city_graph.pkl"
        graph_project_file = "data/Amherst_city_graph_projected.pkl"  # "Amherst_city_graph_projected.pkl"
        graph_project, graph_orig = controller_obj.read_map_data(False, graph_origin_file, graph_project_file)
        source = ox.get_nearest_node(graph_orig, (src_lat_long))
        destination = ox.get_nearest_node(graph_orig, (destination_lat_long))
        
        print ("graph_project.source: ", source)
        print ("graph_project.dst: ", destination)
        route1 = self.ground_truth_shorest_route(graph_orig, source=source, destination=destination, weight='length')
            
        shortest_path_length = self.get_ground_truth_shorest_length(graph_orig, source, destination)       # self.get_total_length(graph_projection, shortest_path)
        overhead = 50
        allowed_cost = ((100.0 + overhead)*shortest_path_length)/100.0
        
        elevation_mode = "minimize"
        heuristic = 'straight-line' #'shortest-path'
        route4 = self.get_A_star_evelation_shorest_perentage_route(graph_orig, source, destination, allowed_cost, heuristic=heuristic, elevation_mode=elevation_mode)
       
        self.plot_two_routes(graph_orig, route1, route4, src_lat_long, destination_lat_long)


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
        
    def run(self, start, end, ele_mode, cost_percentage):
        start = ox.geocode(start)
        target = ox.geocode(end)

        graph_origin_file = "data/Amherst_city_graph.pkl"
        graph_project_file = "data/Amherst_city_graph_projected.pkl"  # "Amherst_city_graph_projected.pkl"
        graph_project, graph_orig = controller_obj.read_map_data(False, graph_origin_file, graph_project_file)
        source = ox.get_nearest_node(graph_orig, (start))
        destination = ox.get_nearest_node(graph_orig, (target))

        shortest_path_length = self.get_ground_truth_shorest_length(graph_orig, source,
                                                                    destination)  # self.get_total_length(graph_projection, shortest_path)
        overhead = cost_percentage
        allowed_cost = ((100.0 + overhead) * shortest_path_length) / 100.0

        elevation_mode = ele_mode
        route2 = self.get_dijkstra_evelation_shorest_perentage_route(graph_orig, source, destination, allowed_cost,
                                                                     elevation_mode=elevation_mode)
        x = ox.plot_route_folium(graph_orig, route2, route_color='green')

        folium.Marker(location=start,
                      icon=folium.Icon(color='red')).add_to(x)
        folium.Marker(location=target,
                      icon=folium.Icon(color='blue')).add_to(x)
        filepath = "output/example.html"
        x.save(filepath)
        #webbrowser.open(filepath, new=2)
    
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
    from_ = "umass amherst"
    to_ = "one east pleasant st amherst ma 01002"
    #controller_obj.run(from_, to_, "maximize", 50)
    
    #controller_obj.test_dijkstra()
    controller_obj.test_Atar()    
    #controller_obj.test2()
