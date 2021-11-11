# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:06:39 2021

@author: littl
"""
import networkx as nx
from random import choice, random, seed
from collections import Counter
import os
import ray
import pandas as pd
from operator import attrgetter
from pathlib import Path
from copy import deepcopy
import sys
import time

def main():
    start_time = time.time()
    #Input
    global input_dir
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    graphs = get_graphs(input_dir)
    #Set parameters as global variables
    global population_size, max_trials, max_iterations, onlooker_bees, d, best_n
    global beta, lmbda, mu1, mu2, small_value
    
    population_size = 50
    max_trials = 25
    max_iterations = 30
    onlooker_bees = 20
    d = 5
    best_n = int(round(population_size/d))
    
    beta = 0.2
    lmbda = 0.6
    mu1 = 0.5
    mu2 = 0.5
    small_value = 0.05
    
    ray.init(num_cpus=16)
    global mds
    mds = get_mds(graphs)
    #ray.shutdown()
    isol_nodes = get_isolated_nodes(graphs)
    double_list = get_doublets(graphs)
    remove_doublets(graphs, mds)
    analyze(graphs, mds, isol_nodes, double_list, output_dir)
    total_time = time.time() - start_time
    print("Finished in {}".format(total_time))

class solution:
    
    def __init__(self, graph, nodes = set()):
        self.graph = graph
        self.edge_list = nx.to_pandas_edgelist(self.graph, source = 'Parm1', target = 'Parm2')
        self.nodes = set(nodes)
        self.parent_nodes = self.nodes
        self.fitness = self.fitness_score()
        
        
    def init(self):
        seed()
        while nx.is_dominating_set(self.graph, self.nodes) == False:
            self.nodes.add(choice(list(self.graph.nodes())))
        self.update_fitness()
        
    def fitness_score(self):
        fitness_score = self.edge_list.loc[self.edge_list['Parm2'].isin(list(self.nodes)) + self.edge_list['Parm1'].isin(list(self.nodes)), 'weight'].sum()
        return fitness_score
    
    def update_fitness(self):
        self.fitness = self.fitness_score()
    
    def repair(self):
        while nx.is_dominating_set(self.graph, self.nodes) == False:
            self.nodes.add(self.random_heuristic_node())
        self.update_fitness()
        
    def random_heuristic_node(self):
        random_choice = choice([1,2,3])
        heuristics = {1:{}, 2:{}, 3:{}}
        V = [node for node in list(self.graph.nodes()) if node not in self.nodes]
        vertex = {}
        neighbors_of_set = set()
        for node in self.nodes:
            neighbors_of_node = set(self.graph.neighbors(node))
            for neighbor in neighbors_of_node:
                neighbors_of_set.add(neighbor)
        dominated = set.union(self.nodes, neighbors_of_set)
        for node in V:
            vertex[node] = {}
            undominated_neighbors = [node for node in list(self.graph.neighbors(node)) if node not in dominated]
            vertex[node]['weight_sum'] = self.graph.degree(node, 'weight')
            vertex[node]['degree'] = self.graph.degree(node)
            vertex[node]['weight_sum_of_non_dominated_neighbors'] = 0
            for neighbor in undominated_neighbors:
                vertex[node]['weight_sum_of_non_dominated_neighbors'] += self.graph.degree(neighbor, 'weight')
        for node in vertex:
            heuristics[1][node] = vertex[node]['degree']/vertex[node]['weight_sum']
            heuristics[2][node] = vertex[node]['weight_sum_of_non_dominated_neighbors']/vertex[node]['weight_sum']
            heuristics[3][node] = (vertex[node]['weight_sum_of_non_dominated_neighbors']*vertex[node]['degree'])/vertex[node]['weight_sum']
        return max(heuristics[random_choice], key=heuristics[random_choice].get)
    
    def improve(self):
        redundant_nodes = []
        for node in self.nodes:
            tmp_set = deepcopy(self.nodes)
            tmp_set.remove(node)
            if nx.is_dominating_set(self.graph, tmp_set) == True:
                redundant_nodes.append(node)
        while redundant_nodes != []:
            self.nodes.remove(self.heuristic_improv(redundant_nodes))
            redundant_nodes = []
            for node in redundant_nodes:
                tmp_set = deepcopy(self.nodes)
                tmp_set.remove(node)
                if nx.is_dominating_set(self.graph, tmp_set) == True:
                    redundant_nodes.append(node)
        self.update_fitness()
    
    def heuristic_improv(self, redundant_nodes):
        heuristics = {}
        vertex = {}
        for node in redundant_nodes:
            vertex[node] = {}
            vertex[node]['weight_sum'] = self.graph.degree(node, 'weight')
            vertex[node]['degree'] = self.graph.degree(node)
        for node in vertex:
            heuristics[node] = vertex[node]['weight_sum']/vertex[node]['degree']
        return max(heuristics, key=heuristics.get)
    
    def mutate(self, probability_vector):
        seed()
        all_nodes = list(self.graph.nodes())
        mutation = set()
        for node in all_nodes:
            if random() < beta and random() < probability_vector[node]:
                mutation.add(node)
            elif node in self.nodes:
                mutation.add(node)
        return mutation 
    
    def emp_bee_phase(self, probability_vector):
        trials_with_no_change = 0
        mutation = solution(self.graph, self.mutate(probability_vector))
        if mutation.fitness_score() < self.fitness_score():
            self.nodes = deepcopy(mutation.nodes)
            self.update_fitness()
        else:
            trials_with_no_change += 1
        while trials_with_no_change <= max_trials:
            mutation = solution(self.graph, self.mutate(probability_vector))
            if mutation.fitness_score() < self.fitness_score():
                self.nodes = deepcopy(mutation.nodes)
                self.update_fitness()
            else:
                trials_with_no_change += 1
    
    def scout_bee_phase(self, best_iteration):
        seed()
        if random() < mu1:
            new_solution = solution(self.graph)
            new_solution.init()
            self = new_solution
        else:
            new_solution = solution(self.graph)
            for node in best_iteration.nodes:
                if random() < mu2:
                    new_solution.nodes.add(node)
            new_solution.repair()
            self = deepcopy(new_solution)

@ray.remote            
def abc_eda(graph):
    seed()
    solution_list = init_pop(graph)
    best_solution = best(solution_list)
    best_iteration = deepcopy(best_solution)
    probability_vector = init_prob_v(solution_list, graph)
    iterations = 0
    while iterations <= max_iterations:
        iterations += 1
        for sol in solution_list:
            sol.emp_bee_phase(probability_vector)
            if sol.fitness_score() < best_iteration.fitness_score():
                best_iteration = deepcopy(sol)
            sol.scout_bee_phase(best_iteration)
        onlooker_bee_phase(graph, probability_vector, solution_list, best_iteration)
        update_prob_v(probability_vector, solution_list)
        if best_iteration.fitness_score() < best_solution.fitness_score():
            best_solution = best_iteration
    return list(best_solution.nodes)

def init_pop(graph):
    first_solution = solution(graph)
    first_solution.repair()
    second_solution = solution(graph, list(graph.nodes()))
    second_solution.improve()
    rest = [solution(graph) for i in range(2,population_size)]
    for sol in rest:
        sol.init()
    solution_list = [first_solution, second_solution] + rest
    return solution_list

def best(solution_list):
    best_solution = min(solution_list,key=attrgetter('fitness'))
    return best_solution

def init_prob_v(solution_list, graph):
    all_nodes = list(graph.nodes())
    best_n_solutions = sorted(solution_list,key=attrgetter('fitness'))[:best_n]
    list_of_occurence = []
    for sol in best_n_solutions:
        list_of_occurence += sol.nodes
    probability_vector = dict(Counter(list_of_occurence))
    for node in probability_vector:
        probability_vector[node] = probability_vector[node]/best_n
    for node in all_nodes:
        if node not in probability_vector.keys():
            probability_vector[node] = small_value
    return probability_vector

def update_prob_v(probability_vector, solution_list):
    best_n_solutions = sorted(solution_list,key=attrgetter('fitness'))[:best_n]
    list_of_occurence = []
    for sol in best_n_solutions:
        list_of_occurence += sol.nodes
    occurences = dict(Counter(list_of_occurence))
    for node in occurences:
        probability_vector[node] = (1-lmbda)*probability_vector[node] + lmbda*(occurences[node]/best_n)

def onlooker_bee_phase(graph, probability_vector, solution_list, best_iteration):
    seed()
    for bee in range(onlooker_bees):
        index = choice(range(len(solution_list)))
        new_solution = solution(graph, solution_list[index].mutate(probability_vector))
        new_solution.repair()
        new_solution.improve()
        if new_solution.fitness_score() < solution_list[index].fitness_score():
            solution_list[index] = deepcopy(new_solution)
            if new_solution.fitness_score() < best_iteration.fitness_score():
                best_iteration = deepcopy(new_solution)
        return best_iteration
        
def get_mds(graphs):
    mds_dict = {graph: abc_eda.remote(graphs[graph]) for graph in graphs}
    mds_dict = {graph: ray.get(mds_dict[graph]) for graph in graphs}
    return mds_dict        

###### Output Processing #####

def get_graphs(directory):
    graphs = {}
    for i in range(0,len(os.listdir(directory))): #iterate over files in directory
        if os.listdir(directory)[i].endswith(".csv"):
            graph_name = os.path.splitext(os.listdir(directory)[i])[0] #set name of graph as filename without extension
            edge_list = pd.read_csv(str(directory + "/" + os.listdir(directory)[i])) #import edge list
            edge_list = edge_list[edge_list['Parm1'] != edge_list['Parm2']]
            G = nx.from_pandas_edgelist(edge_list, source = 'Parm1', target = 'Parm2', edge_attr = 'weight')
            graphs[graph_name] = G
    return graphs

def get_doublets(graphs):
    double_list = {}
    for graph in graphs:
        double_list[graph] = []
        for component in nx.connected_components(graphs[graph]):
            if len(component) == 2:
                double_list[graph].append(component)
    return double_list

def remove_doublets(graphs,mds):
    for graph in graphs:
        for component in nx.connected_components(graphs[graph]):
            if len(component) == 2:
                for node in component:
                    if node in mds[graph]:
                        mds[graph].remove(node)

def get_isolated_nodes(graphs):
    isol_nodes = {}
    for i in graphs:
        isol_nodes[i] = list(nx.isolates(graphs[i]))
    return isol_nodes

def analyze(graphs, ds, isol_nodes, double_list, output_dir):
    Path(output_dir).mkdir(parents = True, exist_ok = True, mode = 0o666)
    with open(output_dir + '/{path}_ds.csv'.format(path = input_dir), 'w') as file:
        file.write("subset_threshold, dominating_set, doublets, isolated_nodes\n")
        for graph in ds.keys():
            ds_list = str(ds[graph])
            ds_list = "\"" + ds_list[1: len(ds_list) - 1] + "\""
            doubles_list = str(double_list[graph])
            doubles_list = "\"" + doubles_list[1: len(doubles_list) - 1] + "\""
            isol_list = str(isol_nodes[graph])
            isol_list = "\"" + isol_list[1: len(isol_list) - 1] + "\""
            file.write("%s,%s,%s,%s\n"%(graph,ds_list, doubles_list, isol_list))


if __name__ == '__main__':
    main()

