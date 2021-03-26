#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import scipy.sparse as spa
import cvxpy as cp
from math import radians, cos, sin, asin, sqrt

class TP:
    
    def __init__(self, data, taxis=2, taxi_lon=-73.9772, taxi_lat=40.7527, mph=20, driving_cost=.6, taxi_cost=0, flex=20):
        
        n = len(data.index)
        source = n
        sink = n+1
        num_nodes = n+2
        
        year = data.loc[0, 'pickup_datetime'].year
        month = data.loc[0, 'pickup_datetime'].month
        day = data.loc[0, 'pickup_datetime'].day
        midnight = datetime.datetime(year, month, day, 0, 0)
        flex = timedelta(minutes=flex)
        
        possible = np.zeros((num_nodes, num_nodes))  # =1 if connection (i,j) is possible, =0 if impossible
        profits = np.zeros((num_nodes, num_nodes))   # Net profit from accepting passenger j (based on preceding passenger i)
                                                     # Note: =0 if connection (i,j) is impossible
    
        for i in range(n):
            possible[i, sink] = 1
#             dist_between = self.haversine(data.loc[i, 'dropoff_longitude'], data.loc[i, 'dropoff_latitude'], taxi_lon, taxi_lat)
#             profits[i, sink] = -driving_cost * dist_between
            for j in range(n):
                if (i == j): continue
                dist_between = self.haversine(data.loc[i, 'dropoff_longitude'], data.loc[i, 'dropoff_latitude'],
                                              data.loc[j, 'pickup_longitude'], data.loc[j, 'pickup_latitude'])
                time_between = timedelta(hours = dist_between / mph)
                if (data.loc[i, 'dropoff_datetime'] + time_between > data.loc[j, 'pickup_datetime'] + flex): continue
                else: possible[i,j] = 1
                profits[i,j] = data.loc[j, 'total_amount'] - driving_cost * (dist_between + data.loc[j, 'trip_distance'])

                # Taxis (Source to First Customers)
                dist_between = self.haversine(taxi_lon, taxi_lat, data.loc[j, 'pickup_longitude'], data.loc[j, 'pickup_latitude'])
                time_between = timedelta(hours = dist_between / mph)
                if (midnight + time_between > data.loc[j, 'pickup_datetime'] + flex): continue
                else: possible[source,j] = 1
                profits[source,j] = data.loc[j, 'total_amount'] - driving_cost * (dist_between + data.loc[j, 'trip_distance']) - taxi_cost
        
        arcs = {}
        a = 0
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if (possible[i,j] == 1):
                    arcs[a] = (i,j)
                    a += 1
                if (possible[j,i] == 1):
                    arcs[a] = (j,i)
                    a += 1
        
        self.data = data
        self.n = n
        self.source = source
        self.sink = sink
        self.num_nodes = num_nodes
        self.num_taxis = taxis
        self.taxi_lon = taxi_lon
        self.taxi_lat = taxi_lat
        self.mph = mph
        self.midnight = midnight
        self.profits = profits
        self.arcs = arcs
        self.num_arcs = len(arcs)
        
    # Set-Up Functions for Optimization Problems
        
    def flow_vars(self):
        A_in = spa.dok_matrix((self.num_nodes, self.num_arcs))
        A_out = spa.dok_matrix((self.num_nodes, self.num_arcs))
        p = np.zeros(self.num_arcs)
        for a in range(self.num_arcs):
            i = self.arcs[a][0]
            j = self.arcs[a][1]
            A_in[i,a] = 1
            A_out[j,a] = 1
            p[a] = self.profits[i,j]
        
        e = np.zeros(self.num_nodes)
        e[self.source] = self.num_taxis
        e[self.sink] = -self.num_taxis
        
        return A_in, A_out, e, p
    
    def time_vars(self, time_window):
        B = spa.dok_matrix((self.num_arcs, self.num_arcs))
        d = np.zeros(self.num_arcs)
        t_min, t_max = self.time_cons(time_window)
        for a in range(self.num_arcs):
            i = self.arcs[a][0]
            j = self.arcs[a][1]
            B[a,a] = t_max[i] - t_min[j] + self.T(i,j)
            d[a] = t_max[i] - t_min[j]
        return B, d
    
    def time_cons(self, time_window):
        t_min = []
        t_max = []
        time_window = timedelta(minutes=time_window)
        for i in range(self.n):
            t_min.append(self.to_minutes(self.data.loc[i, 'pickup_datetime'] - self.midnight))
            t_max.append(self.to_minutes(self.data.loc[i, 'pickup_datetime'] - self.midnight + time_window))
        for k in range(self.n, self.num_nodes):
            t_min.append(0)
            t_max.append(0)
#         hours_24 = timedelta(hours=24)
#         t_min.append(self.to_minutes(self.midnight + hours_24))
#         t_max.append(self.to_minutes(self.midnight + hours_24))
        return t_min, t_max
        
    def T(self, i, j):
        if (i == j or i == self.source): return 0
        # From customer to sink
        if (j == self.sink): return -9999999
        # From taxi to customer
        if (i >= self.n):
            dist_between = self.haversine(self.taxi_lon, self.taxi_lat, self.data.loc[j, 'pickup_longitude'], self.data.loc[j, 'pickup_latitude'])
            output = timedelta(hours = dist_between / self.mph)
        # From customer to customer
        else:
            dist_between = self.haversine(self.data.loc[i,'dropoff_longitude'], self.data.loc[i, 'dropoff_latitude'],
                                          self.data.loc[j, 'pickup_longitude'], self.data.loc[j, 'pickup_latitude'])
            time_between = timedelta(hours = dist_between / self.mph)
            output = self.data.loc[i, 'dropoff_datetime'] - self.data.loc[i, 'pickup_datetime'] + time_between
        return self.to_minutes(output)
    
    # Problem with Time Windows
    
    def problem_window(self, time_window=0):
        # Variables
        x = cp.Variable(self.num_arcs, boolean=True)
        f = cp.Variable(self.n)
        t = cp.Variable(self.n)
        
        A_in, A_out, e, p = self.flow_vars()
        A = A_in - A_out
        B, d = self.time_vars(time_window)
        C = np.transpose(A[:self.n,])
        
        ## Network Flow Constraints
        constraints = [A @ x <= e, f >= 0, f <= 1]
        inflow = A_in @ x
        outflow = A_out @ x
        constraints += [inflow[:self.n] == f, outflow[:self.n] == f]

        ## Time Window Constraints
        t_min, t_max = self.time_cons(time_window)
        constraints += [t_min[:self.n] <= t, t <= t_max[:self.n]]
        constraints += [B @ x + C @ t <= d]

        profit = np.transpose(p) @ x
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return x, problem
    
    # Problem with Times as Parameters
    
    def time_params(self, t, time_window):
        b = []
        d = []
#         d = cp.Variable(self.num_arcs)
        time_cons = []
        t_min = t
        t_max = t + time_window
        for a in range(self.num_arcs):
            i = self.arcs[a][0]
            j = self.arcs[a][1]
            b.append(t_max[i] - t_min[j] + self.T(i,j))
            d.append(t_max[i] - t_min[j])
#             time_cons += [d[a] == t_max[i] - t_min[j]]
        return cp.hstack(b), cp.hstack(d), time_cons
    
    def problem_param(self, time_window=0):
        # Variables and Parameter
        x = cp.Variable(self.num_arcs)
        f = cp.Variable(self.n)
        t = cp.Parameter(self.num_nodes)

        A_in, A_out, e, p = self.flow_vars()
        A = A_in - A_out
        b, d, constraints = self.time_params(t, time_window)
        C = np.transpose(A[:self.n,])
        
        constraints += [A @ x <= e, x >= 0, x <= 1, f >= 0, f <= 1]
        inflow = A_in @ x
        outflow = A_out @ x
        constraints += [inflow[:self.n] == f, outflow[:self.n] == f]
        constraints += [cp.multiply(b,x) + C @ t[:self.n] <= d]
        
        profit = np.transpose(p) @ x
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return t, x, problem
    
    # Helper Functions
    
    def to_minutes(self, td):
        return td.total_seconds() / 60

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 3956
        return c * r