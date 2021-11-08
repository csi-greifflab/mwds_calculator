# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:35:05 2021

@author: Jahn Zhong
"""
"""
Probability parameters:
    
beta:               proportion of nodes sampled from probability vector instead of previous generation for guided mutation
lmbda:              
small_value:        probability of nodes not counted when initializing probability vector
mu1:                probability of completely abandoning solution and generating a new one 
mu2:                closeness of new solution during scout bee phase if solution not completely abandoned
-----------------------------------------------------------------------------------------------------------------------------
Other parameters

max_iterations:     Iteration number of main algorithm; has biggest impact on solution quality
population_size:    number of random solutions to evaluate
d:                  population_size/d best solutions used to generate and update probability distribution vector
max_trials:         Higher number --> Deeper solutions exploration in employed bee phase
onlooker_bees:      Number of bees generating neighboring solutions from previous emplyed bee phase

"""


import networkx as nx
from random import choice, random, seed
from collections import Counter
import os
import ray
import numpy
import pandas as pd
from operator import itemgetter
from pathlib import Path

def main():                  
    #Setup
    ray.init(num_cpus=8)
    input_dir = 'test'
    graphs = get_graphs(input_dir)
    graphs_with_isolated = get_graph_with_isolated(input_dir)
    isolated_nodes = get_isolated_nodes(graphs_with_isolated)
    mds_dict = get_mds(graphs)
    doublets = get_doublets(graphs, mds_dict)
    remove_doublets(graphs, mds_dict)
    analyze(graphs, mds_dict, isolated_nodes, doublets, input_dir)
    ray.shutdown()

def analyze(graphs, ds, isol_nodes, double_list, directory):
    output_dir = "output"
    Path(output_dir).mkdir(parents = True, exist_ok = True, mode = 0o666)
    with open(output_dir + '{path}_ds.csv'.format(path = directory), 'w') as file:
        file.write("subset_threshold, dominating_set, doublets, isolated_nodes\n")
        for graph in ds.keys():
            ds_list = str(ds[graph])
            ds_list = "\"" + ds_list[1: len(ds_list) - 1] + "\""
            doubles_list = str(double_list[graph])
            doubles_list = "\"" + doubles_list[1: len(doubles_list) - 1] + "\""
            isol_list = str(isol_nodes[graph])
            isol_list = "\"" + isol_list[1: len(isol_list) - 1] + "\""
            file.write("%s,%s,%s,%s\n"%(graph,ds_list, doubles_list, isol_list))

def get_subgraphs(graphs, edge_lists):
    subgraphs = {}
    component_sets = {}
    for i in graphs:
        component_sets[i] = {}
        num = 1
        for j in nx.connected_components(graphs[i]):
            name = 'sub' + str(num)
            if len(j) > 2:
                component_sets[i][name] = j
                num += 1
    for k in component_sets:
        subgraphs[k] = {}
        for l in component_sets[k]:
            subgraphs[k][l] = graphs[k].subgraph(nodes = component_sets[k][l])
    return subgraphs

def get_ds_with_isol(ds, isol):
    ds_with_isol = {}
    for key in ds:
        ds_with_isol[key] = []
        for node in ds[key]:
            ds_with_isol[key].append(node)
        for node in isol[key]:
            ds_with_isol[key].append(node)
    return ds_with_isol

def get_doublets(graphs, ds):
    doublets = {}
    for graph in graphs:
        doublets[graph] = []
        for component in nx.connected_components(graphs[graph]):
            if len(component) == 2:
                doublets[graph].append(component)
    return doublets

def remove_doublets(graphs,ds):
    for graph in graphs:
        for component in nx.connected_components(graphs[graph]):
            if len(component) == 2:
                for node in component:
                    if node in ds[graph]:
                        ds[graph].remove(node)
    
def get_edge_lists(graphs):
    if type(graphs) is dict:
        edge_lists = {graph: nx.to_pandas_edgelist(graphs[graph], source = 'Parm1', target = 'Parm2') for graph in graphs}
        return edge_lists
    else:
        edge_list = nx.to_pandas_edgelist(graphs, source = 'Parm1', target = 'Parm2')
        return edge_list

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

def get_graph_with_isolated(directory):
    graphs = {}
    for i in range(0,len(os.listdir(directory))): #iterate over files in directory
        if os.listdir(directory)[i].endswith(".csv"):
            graph_name = os.path.splitext(os.listdir(directory)[i])[0] #set name of graph as filename without extension
            edge_list = pd.read_csv(str(directory + "/" + os.listdir(directory)[i])) #import edge list
            G = nx.from_pandas_edgelist(edge_list, source = 'Parm1', target = 'Parm2', edge_attr = 'weight')
            G.remove_edges_from(nx.selfloop_edges(G))
            graphs[graph_name] = G
    return graphs

def get_isolated_nodes(graphs):
    isol_nodes = {}
    for i in graphs:
        isol_nodes[i] = list(nx.isolates(graphs[i]))
    return isol_nodes

@ray.remote
def abc_eda(
        graph, edge_list, population_size, max_iterations, max_trials, 
        onlooker_bees,  d, beta, lmbda, mu1, mu2, small_value):
    seed()
    best_n = int(round(population_size/d))
    population, best_solution = init_pop(graph, population_size, edge_list)
    best_iteration = best_solution.copy()
    probability_vector = init_prob_v(graph, population, best_n, small_value)
    iterations = 1
    while iterations <= max_iterations:
        iterations += 1
        population, best_iteration = emp_bee_phase(graph, population, edge_list, max_trials, best_iteration, probability_vector, mu1, mu2, population_size, beta)
        population, best_iteration = onlooker_bee_phase(population, graph, edge_list, probability_vector, best_iteration, onlooker_bees, beta)
        probability_vector = update_prob_v(population, probability_vector, edge_list, lmbda, best_n)
        if fitness(best_iteration, edge_list) < fitness(best_solution, edge_list):
            best_solution = best_iteration
    return list(set(best_solution))

def guided_mutation(
        graph, 
        solution, 
        probability_vector, beta):
    V = list(graph.nodes())
    mutation = []
    for node in V:
        if random() < beta:
            if random() < probability_vector[node]:
                mutation.append(node)
        elif node in solution:
            mutation.append(node)
    return mutation
    
def init_prob_v(graph, solutions, best_n, small_value):
    V = list(graph.nodes())
    solution_quality = {key: fitness(solutions[key], get_edge_lists(graph)) for key, value in solutions.items()}
    tmp_dict = dict(sorted(solution_quality.items(),key = itemgetter(1))[:best_n])
    best_n_solutions = {key: solutions[key] for key in tmp_dict.keys()}
    list_of_occurence = []
    for solution in best_n_solutions:
        list_of_occurence += best_n_solutions[solution]
    probability_vector = dict(Counter(list_of_occurence))
    for node in probability_vector:
        probability_vector[node] = probability_vector[node]/best_n
    for node in V:
        if node not in probability_vector.keys():
            probability_vector[node] = float(small_value)
    return probability_vector

#Updates probability vector based on 
def update_prob_v(solutions, probability_vector, edge_list, lmbda, best_n):
    solution_quality = {key: fitness(solutions[key], edge_list) for key, value in solutions.items()}
    tmp_dict = dict(sorted(solution_quality.items(),key = itemgetter(1))[:best_n])
    best_n_solutions = {key: solutions[key] for key in tmp_dict.keys()}
    list_of_occurence = []
    for solution in best_n_solutions:
        list_of_occurence += best_n_solutions[solution]
    occurences = dict(Counter(list_of_occurence))
    for node in probability_vector:
        if node in occurences.keys():
            probability_vector[node] =  (1-lmbda)*probability_vector[node] + lmbda*(occurences[node]/best_n)
    return probability_vector

#Evaluates fitness by sum of edge weights associated                                                                                     
def fitness(solution, edge_list):
    fitness_score = edge_list.loc[edge_list['Parm2'].isin(solution) + edge_list['Parm1'].isin(solution), 'weight'].sum()
    return fitness_score

#Adds nodes to solution until solution is dominating set
def repair(graph, solution):
    while nx.is_dominating_set(graph, solution) == False:
        solution.append(random_heuristic_node(graph, solution))

#Removes redundant nodes
def improve(graph, solution):
    redundant_nodes = []
    for node in solution:
        tmp = solution.copy()
        tmp.remove(node)
        if nx.is_dominating_set(graph, tmp) == True:
            redundant_nodes.append(node)
    while redundant_nodes != []:
        solution.remove(heuristic_improv(graph, redundant_nodes))
        redundant_nodes = []
        for node in solution:
            tmp = solution.copy()
            tmp.remove(node)
            if nx.is_dominating_set(graph, tmp) == True:
                R.append(node)

#generates solution by randomly adding nodes
def init(graph):
    solution = set()
    remaining_nodes = set(graph.nodes())
    while nx.is_dominating_set(graph, solution) == False:
        random_node = choice(list(remaining_nodes))
        solution.add(random_node)
        remaining_nodes.remove(random_node)        
    return list(solution)

#Generates initial solutions and returns best solution among them  
def init_pop(graph, population_size, edge_list):
    solutions = {i: init(graph) for i in range(2, population_size)}
    solutions[0] = []
    repair(graph, solutions[0])
    solutions[1] = list(graph.nodes())
    improve(graph, solutions[1])
    fitness_scores = {solution: fitness(solutions[solution], edge_list) for solution in solutions}
    best_solution = solutions[min(fitness_scores, key=fitness_scores.get)]
    return solutions, best_solution

def emp_bee_phase(
        graph, 
        solutions, 
        edge_list, 
        max_trials, 
        best_iteration, 
        probability_vector, mu1, mu2, population_size, beta):
    trials_with_no_change = 0
    for i in range(0,population_size):
        index = i
        trials_with_no_change += 1
        mutated_solution = guided_mutation(graph, solutions[index], probability_vector, beta)
        repair(graph, mutated_solution)
        improve(graph, mutated_solution)
        if fitness(mutated_solution, edge_list) < fitness(solutions[index], edge_list):
            solutions[index] = mutated_solution
            trials_with_no_change = 0
        elif trials_with_no_change == max_trials:
            solutions = scout_bee_phase(graph, solutions, index, best_iteration, mu1, mu2)
        if fitness(solutions[index], edge_list) < fitness(best_iteration, edge_list):
            best_iteration = solutions[index]
    return solutions, best_iteration

def scout_bee_phase(graph, solutions, index, best_iteration, mu1, mu2):
    if random() < mu1:
        solutions[index] = init(graph)
    else:
        solutions[index] = []
        for node in best_iteration:
            if random() < mu2:
                solutions[index].append(node)
        repair(graph, solutions[index])
    return solutions

def onlooker_bee_phase(
        solutions, graph, edge_list, probability_vector, best_iteration, onlooker_bees, beta):
    for i in range(0,onlooker_bees):
        index = choice(range(0,len(solutions)))
        O = guided_mutation(graph, solutions[index], probability_vector, beta)
        repair(graph, O)
        improve(graph, O)
        if fitness(O, edge_list) < fitness(best_iteration, edge_list):
            best_iteration = O
        if fitness(O, edge_list) < fitness(solutions[index], edge_list):
            solutions[index] = O
    return solutions, best_iteration


def random_heuristic_node(graph, solution):
    random_choice = choice([1,2,3])
    heuristics = {1:{}, 2:{}, 3:{}}
    V = [node for node in list(graph.nodes()) if node not in solution]
    vertex = {}
    dominated = []
    dominated += solution
    for node in solution:
        dominated += list(graph.neighbors(node))
    dominated = list(set(dominated))
    for node in V:
        vertex[node] = {}
        undominated_neighbors = [node for node in list(graph.neighbors(node)) if node not in dominated]
        vertex[node]['weight_sum'] = graph.degree(node, 'weight')
        vertex[node]['degree'] = graph.degree(node)
        vertex[node]['weight_sum_of_non_dominated_neighbors'] = 0
        for neighbor in undominated_neighbors:
            vertex[node]['weight_sum_of_non_dominated_neighbors'] += graph[node][neighbor]['weight']
    for node in vertex:
        heuristics[1][node] = vertex[node]['degree']/vertex[node]['weight_sum']
        heuristics[2][node] = vertex[node]['weight_sum_of_non_dominated_neighbors']/vertex[node]['weight_sum']
        heuristics[3][node] = (vertex[node]['weight_sum_of_non_dominated_neighbors']*vertex[node]['degree'])/vertex[node]['weight_sum']
    return max(heuristics[random_choice], key=heuristics[random_choice].get)

def heuristic_improv(graph, solution):
    heuristics = {1:{}}
    V = solution
    vertex = {}
    for node in V:
        vertex[node] = {}
        vertex[node]['weight_sum'] = graph.degree(node, 'weight')
        vertex[node]['degree'] = graph.degree(node)
    for node in vertex:
        heuristics[1][node] = vertex[node]['weight_sum']/vertex[node]['degree']
    return max(heuristics[1], key=heuristics[1].get)

def test_output(test_n, edge_list):
    output = {}
    overlaps = []
    for i in range(test_n):
        output[i] = abc_eda.remote(graph, edge_list, population_size, max_iterations, max_trials)
    for i in range(test_n):
        output[i] = ray.get(output[i])
    for i in range(test_n):
        for j in range(test_n):
            if i != j:
                overlaps.append(overlap(output[i], output[j]))
    fitness_scores = [fitness(output[i], edge_list) for i in range(test_n)]
    average_fitness = average(fitness_scores)
    std_fitness = numpy.std(fitness_scores)
    average_overlap = average(overlaps)
    std_overlap = numpy.std(overlaps)
    return output, average_overlap, std_overlap, average_fitness, std_fitness

def intersection(set1, set2):
    intersect = [vertex for vertex in set1 if vertex in set2]
    return intersect

def overlap(set1, set2):
    return len(intersection(set1,set2))/((len(set1)+len(set2))/2)

def average(lst):
    return sum(lst)/len(lst)

def get_mds(
        graphs, population_size = 50, max_iterations = 20, max_trials = 25, 
        onlooker_bees = 20, d = 5, beta = 0.2, lmbda = 0.6, mu1 = 0.5, mu2 = 0.5, 
        small_value = 0.05):
    edge_lists = get_edge_lists(graphs)
    mds_dict = {graph: abc_eda.remote(
                        graphs[graph], edge_lists[graph], population_size, max_iterations, max_trials,
                        onlooker_bees,  d, beta, lmbda, mu1, mu2, small_value)
        for graph in graphs}
    mds_dict = {graph: ray.get(mds_dict[graph]) for graph in graphs}
    return mds_dict
  
if __name__ == "__main__":
    main()
