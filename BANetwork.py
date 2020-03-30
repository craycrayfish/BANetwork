#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:38:33 2020

@author: shawn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import os
import networkx as nx
from collections import Counter

class Network:
    def __init__(self, m, prob, N=False):
        '''Creates a graph represented by a list of N lists where each
        inner list shows all neighbours connected to that vertex. Initialises
        with each vertex having 1 edge.
        Parameters
        ----------
        N : int
            Size of graph to initialise
        m : int
            Number of edges to add with each new vertex
        prob : str
            Probability model to use. 'preferential' or 'random'
        '''
        prob_models = ['preferential', 'random']
        if prob not in list(prob_models):
            raise ValueError("prob should be in {}".format(str(prob_models)))
        if N is False:
            self.N = m + 1
        else:
            self.N = N
        self.m = m
        self.prob = prob
        self.graph = self.initialise_graph()
        self.vertex_list = self.initialise_vertex_list()
        self.G = None
        self.vertices = None
        self.degree = None
        self.deg_freq = None
        
    def initialise_graph(self):
        adj_list = [[n] for n in range(self.N)]
        for vertex in adj_list:
            neighbours = [i for i in range(self.N) if i > vertex[0]]
            vertex.extend(neighbours)
        return adj_list
    
    def initialise_vertex_list(self):
        '''List of vertices to choose neighbours from. For the preferential case,
        each vertex is repeated the same number of times as its degree.'''
        vertex_list = [i for i in range(self.N)]
        if self.prob == 'preferential':
            vertex_list *= self.m
        return vertex_list
    
    def convert_networkx(self):
        '''Saves as a networkx graph object'''
        adj_list = [' '.join(map(str, i)) for i in self.graph]
        self.G = nx.parse_adjlist(adj_list)
    
    def draw_graph(self):
        '''Generates the graph using networkx package.'''
        nx.draw(self.G)
    
    def add_vertex(self, neighbours):
        '''Adds a vertex with given neighbours and update neighbours.'''
        for n in neighbours:
            self.graph[n].append(self.N)
        self.graph.append([self.N])
        return self
        
    def add_edges(self, a, b):
        '''Adds an edge between two vertices a and b'''
        self.graph[a].append(b)
        self.graph[b].append(a)
    
    def gen_neighbours(self, m):
        '''Generates a list of m neighbours from vertex list.'''
        neighbours = list(set(random.sample(self.vertex_list, m)))
        i = m - len(neighbours)
        while i > 0:
            neighbours.extend(random.sample(self.vertex_list, i))
            neighbours = list(set(neighbours))
            i = m - len(neighbours)
        return neighbours
    
    def update_vertex_list(self, neighbours):
        '''Adds vertices to a list that new neighbours are 
        drawn from.'''
        if self.prob == 'preferential':
            self.vertex_list.extend(neighbours)
            self.vertex_list.extend([self.N] * self.m)
        elif self.prob == 'random':
            self.vertex_list.append(self.N)
        
    def add_vertices(self, N):
        '''Adds vertices until graph has N number of vertices'''
        if N < self.N:
            print('Given N is smaller than number of vertices')
        while self.N < int(N):
            neighbours = self.gen_neighbours(self.m)
            self.add_vertex(neighbours)
            self.update_vertex_list(neighbours)
            self.N += 1
        return self

    def count_deg(self):
        '''Count the degree for each vertex.'''
        self.vertices = [n for n in range(self.N)]
        self.degree = [0 for n in range(self.N)]
        for vertex in self.graph:
            self.degree[vertex[0]] += len(vertex) - 1
            for neighbour in vertex[1:]:
                self.degree[neighbour] += 1
        return self
    
    def count_deg_freq(self):
        self.deg_freq = dict(Counter(self.degree))
        return self
    

class NetworkRW:
    def __init__(self, m, q, N=False):
        '''Creates a graph represented by a list of N lists where each
        inner list shows all neighbours connected to that vertex. Initialises
        with each vertex having 1 edge.
        Parameters
        ----------
        N : int
            Size of graph to initialise
        m : int
            Number of edges to add with each new vertex
        q : float
            Probability of continuing a random walk after each step
        '''
        if N is False:
            self.N = m + 1
        else:
            self.N = N
        self.m = m
        self.q = q
        self.random = []
        self.graph = self.initialise_graph()
        self.vertices = None
        self.degree = None
        self.deg_freq = None
        
    def initialise_graph(self):
        vertices = [n for n in range(self.N)]
        graph = {}
        for vertex in vertices:
            graph[vertex] = [v for v in vertices if v!=vertex]
        return graph
    
    def add_vertex(self, neighbours):
        '''Adds a vertex with given neighbours and update neighbours.'''
        for n in neighbours:
            self.graph[n].append(self.N)
        self.graph[self.N] = neighbours
        return self
    
    def gen_random(self):
        self.random = np.random.random_sample(2 * self.m / self.q)
    
    def gen_neighbours(self, m):
        '''Generates a list of m neighbours from vertex list.'''
        vertex_list = list(self.graph)
        neighbours = list(set(random.sample(vertex_list, m)))
        rand = np.random.rand
        print('walking for: {}'.format(str(neighbours)))
        for i in range(len(neighbours)):
            print('  start:' + str(neighbours[i]))
            while rand() < self.q:
                neighbours[i] = random.choice(self.graph[neighbours[i]])
                print('    to:' + str(neighbours[i]))
            print('  end:{}'.format(str(neighbours[i])))
        print('end state:{}'.format(str(neighbours)))
        return neighbours
    
    def gen_neighbour(self):
        '''Generates a new neighbour from random walk'''
        rand = np.random.rand()
        neighbour = random.choice(list(self.graph))
        while rand < self.q:
            neighbour = random.choice(self.graph[neighbour])
            rand = np.random.rand()
        return neighbour
    
    def add_vertices(self, N):
        '''Adds vertices until graph has N number of vertices'''
        if N < self.N:
            print('Given N is smaller than number of vertices')
        while self.N < int(N):
            neighbours = []
            while len(neighbours) < self.m:
                neighbour = self.gen_neighbour()
                if neighbour not in neighbours:
                    neighbours.append(neighbour)
            self.add_vertex(neighbours)
            self.N += 1
        return self

    def count_deg(self):
        '''Count the degree for each vertex.'''
        degree = {}
        for vertex in self.graph:
            degree[vertex] = len(self.graph[vertex])
        self.degree = degree
        return self

class NetworkRW:
    def __init__(self, m, q, N=False):
        '''Creates a graph represented by a list of N lists where each
        inner list shows all neighbours connected to that vertex. Initialises
        with each vertex having 1 edge.
        Parameters
        ----------
        N : int
            Size of graph to initialise
        m : int
            Number of edges to add with each new vertex
        prob : str
            Probability model to use. 'preferential' or 'random'
        '''
        prob_models = ['preferential', 'random']
        if N is False:
            self.N = m + 1
        else:
            self.N = N
        self.m = m
        self.q = q
        self.random = []
        self.graph = self.initialise_graph()
        self.vertices = None
        self.degree = None
        self.deg_freq = None
        
    def initialise_graph(self):
        vertices = [n for n in range(self.N)]
        graph = {}
        for vertex in vertices:
            graph[vertex] = [v for v in vertices if v!=vertex]
        return graph
    
    def add_vertex(self, neighbours):
        '''Adds a vertex with given neighbours and update neighbours.'''
        for n in neighbours:
            self.graph[n].append(self.N)
        self.graph[self.N] = neighbours
        return self
    
    def gen_random(self):
        self.random = np.random.random_sample(2 * self.m / self.q)
    
    def gen_neighbours(self, m):
        '''Generates a list of m neighbours from vertex list.'''
        vertex_list = list(self.graph)
        neighbours = list(set(random.sample(vertex_list, m)))
        rand = np.random.rand
        print('walking for: {}'.format(str(neighbours)))
        for i in range(len(neighbours)):
            print('  start:' + str(neighbours[i]))
            while rand() < self.q:
                neighbours[i] = random.choice(self.graph[neighbours[i]])
                print('    to:' + str(neighbours[i]))
            print('  end:{}'.format(str(neighbours[i])))
        print('end state:{}'.format(str(neighbours)))
        return neighbours
    
    def gen_neighbour(self):
        '''Generates a new neighbour from random walk'''
        rand = np.random.rand()
        neighbour = random.choice(list(self.graph))
        while rand < self.q:
            neighbour = random.choice(self.graph[neighbour])
            rand = np.random.rand()
        return neighbour
    
    def add_vertices(self, N):
        '''Adds vertices until graph has N number of vertices'''
        if N < self.N:
            print('Given N is smaller than number of vertices')
        while self.N < int(N):
            neighbours = []
            while len(neighbours) < self.m:
                neighbour = self.gen_neighbour()
                if neighbour not in neighbours:
                    neighbours.append(neighbour)
            self.add_vertex(neighbours)
            self.N += 1
        return self

    def count_deg(self):
        '''Count the degree for each vertex.'''
        degree = {}
        for vertex in self.graph:
            degree[vertex] = len(self.graph[vertex])
        self.degree = degree
        return self