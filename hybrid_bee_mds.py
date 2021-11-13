# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:06:39 2021

@author: Jahn Zhong
"""
import os
import sys
import time
from random import choice, random, seed
from collections import Counter
from operator import attrgetter
from pathlib import Path
from copy import deepcopy

import networkx as nx
import ray
import pandas as pd

import parameters

def main():
    start_time = time.time()
    
    #Input
    global input_dir, output_dir, graphs
    input_dir = Path(os.path.abspath(sys.argv[1]))
    output_dir = Path(os.path.abspath(sys.argv[2]))
    graphs = get_graphs(input_dir)
    
    # Integer parameters
    global population_size, max_trials, max_iterations, onlooker_bees, d, best_n
    population_size = parameters.population_size
    max_trials = parameters.max_trials
    max_iterations = parameters.max_iterations
    onlooker_bees = parameters.onlooker_bees
    d = parameters.d
    best_n = int(round(population_size/d))
    
    # Probability parameters
    global beta, lmbda, mu1, mu2, small_value
    beta = parameters.beta
    lmbda = parameters.lmbda
    mu1 = parameters.mu1
    mu2 = parameters.mu2
    small_value = parameters.small_value
    # Calculate MWDS in parallel
    ray.init(num_cpus=int(sys.argv[3]))
    mds = get_mds()
    ray.shutdown()
    
    # Remove isolated nodes and doublets from mds
    isol_nodes = get_isolated_nodes()
    double_list = get_doublets()
    remove_doublets(mds)
    
    # Save csv with graphs and corresponding mwds, doublets and isolated nodes
    analyze(mds, isol_nodes, double_list)
    
    total_time = time.time() - start_time
    print("Finished in {}".format(total_time))

@ray.remote            
def abc_eda(graph):
    seed()
    # Initialize starting population of solutions and probability_vector
    solution_list = init_pop(graph)
    best_solution = best(solution_list)
    best_iteration = deepcopy(best_solution)
    probability_vector = init_prob_v(solution_list, graph)
    
    # Explore solutions while updating probability_vector after each iteration
    iterations = 0
    while iterations <= max_iterations:
        iterations += 1
        for sol in solution_list:
            sol.emp_bee_phase(probability_vector)
            best_iteration = best(solution_list)
            sol.scout_bee_phase(best_iteration)
        best_iteration, solution_list = onlooker_bee_phase(graph, probability_vector, solution_list)
        probability_vector = update_prob_v(probability_vector, solution_list)
        if best_iteration.fitness_score() < best_solution.fitness_score():
            best_solution = best_iteration
    return list(best_solution.nodes)

class solution:
    
    def __init__(self, graph, nodes = set()):
        self.graph = graph
        self.edge_list = nx.to_pandas_edgelist(self.graph, source = 'Parm1', target = 'Parm2')
        self.nodes = set(nodes)
        self.fitness = self.fitness_score() # lower = better

    def random_solution(self):
        seed()
        self.nodes = set()
        while nx.is_dominating_set(self.graph, self.nodes) == False:
            self.nodes.add(choice(list(self.graph.nodes())))
        self.update_fitness()
        
    def fitness_score(self):
        # sum of weights associated with nodes in solution
        fitness_score = self.edge_list.loc[self.edge_list['Parm2'].isin(list(self.nodes)) + self.edge_list['Parm1'].isin(list(self.nodes)), 'weight'].sum()
        return fitness_score
    
    def update_fitness(self):
        self.fitness = self.fitness_score()

    def repair(self):
        # Adds nodes selected by heuristic functions until entire graph is dominated
        while nx.is_dominating_set(self.graph, self.nodes) == False:
            self.nodes.add(self.random_heuristic_node())
        self.update_fitness()

    def random_heuristic_node(self):
        # Finds all dominated nodes by including the current set and all neighboring nodes
        neighbors_of_set = set()
        for node in self.nodes:
            neighbors_of_node = set(self.graph.neighbors(node))
            for neighbor in neighbors_of_node:
                neighbors_of_set.add(neighbor)
        dominated = set.union(self.nodes, neighbors_of_set)
        
        not_in_solution = [node for node in list(self.graph.nodes()) if node not in self.nodes]
        
        # Store node values in dictionary
        vertex = {}
        for node in not_in_solution:
            vertex[node] = {}
            undominated_neighbors = [node for node in list(self.graph.neighbors(node)) if node not in dominated]
            vertex[node]['weight_sum'] = self.graph.degree(node, 'weight')
            vertex[node]['degree'] = self.graph.degree(node)
            vertex[node]['weight_sum_of_non_dominated_neighbors'] = 0
            for neighbor in undominated_neighbors:
                vertex[node]['weight_sum_of_non_dominated_neighbors'] += self.graph.degree(neighbor, 'weight')
        
        # Calculate best candidate to include in solution in three different ways
        heuristics = {1:{}, 2:{}, 3:{}}
        for node in vertex:
            heuristics[1][node] = vertex[node]['degree']/vertex[node]['weight_sum']
            heuristics[2][node] = vertex[node]['weight_sum_of_non_dominated_neighbors']/vertex[node]['weight_sum']
            heuristics[3][node] = (vertex[node]['weight_sum_of_non_dominated_neighbors']*vertex[node]['degree'])/vertex[node]['weight_sum']
        
        # Randomly selects one of three calculations to use
        random_choice = choice([1,2,3])
        best_node = max(heuristics[random_choice], key=heuristics[random_choice].get)
        return best_node
    
    def improve(self):
        # If a node can be removed from the solution while remaining dominant, it is stored in a list of redundant nodes
        # From this list a node is selected by a heuristic function and removed from the solution
        # Repeat until solution includes no redundant nodes
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
        # Analogous to random_heuristic_node() but with only calculation to determine best candidate for removal
        heuristics = {}
        vertex = {}
        for node in redundant_nodes:
            vertex[node] = {}
            vertex[node]['weight_sum'] = self.graph.degree(node, 'weight')
            vertex[node]['degree'] = self.graph.degree(node)
        heuristics = {node: vertex[node]['weight_sum']/vertex[node]['degree'] for node in vertex.keys()}
        best_node = max(heuristics, key=heuristics.get)
        return best_node
    
    def mutate(self, probability_vector):
        seed()
        all_nodes = list(self.graph.nodes())
        
        # Generate neighboring solution by adding nodes from the parent solution and the probability_vector
        # Parameter beta determines the balance between new nodes and inherited nodes included in the set
        # The higher the beta, the more aggressive.
        mutation = set()
        for node in all_nodes:
            if random() < beta and random() < probability_vector[node]:
                mutation.add(node)
            elif node in self.nodes:
                mutation.add(node)
        return mutation 
    
    def emp_bee_phase(self, probability_vector):
        
        # Mutates current solution until no improvement has been made for max_trials
        trials_with_no_change = 0
        while trials_with_no_change <= max_trials:
            mutation = solution(self.graph, self.mutate(probability_vector))
            if mutation.fitness_score() < self.fitness_score():
                self = deepcopy(mutation)
            else:
                trials_with_no_change += 1

    def scout_bee_phase(self, best_iteration):
        seed()
        # After employed_bee_phase() exhausted local solution space:
        # mu1 is the probability of selecting a new completely random solution
        if random() < mu1:
            self.random_solution()
        # mu2 is the probability of each node in the best solution being used to construct the new solution instead
        else:
            self.nodes = set()
            for node in best_iteration.nodes:
                if random() < mu2:
                    self.nodes.add(node)
            self.repair()

def init_pop(graph):
    # Generates initial population of solutions
    # First solution is an empty repaired solution
    # Second solution is solution containing all nodes that has been improved
    # The rest of the solutions are generated by adding random nodes until valid
    first_solution = solution(graph)
    first_solution.repair()
    second_solution = solution(graph, list(graph.nodes()))
    second_solution.improve()
    rest = [solution(graph) for i in range(2,population_size)]
    for sol in rest:
        sol.random_solution()
    solution_list = [first_solution, second_solution] + rest
    return solution_list

def best(solution_list):
    best_solution = min(solution_list,key=attrgetter('fitness'))
    return best_solution

def init_prob_v(solution_list, graph):
    all_nodes = list(graph.nodes())
    best_n_solutions = sorted(solution_list,key=attrgetter('fitness'))[:best_n]
    
    # Convert frequency of node in best_n solutions to probability and store it probability_vector
    list_of_occurence = []
    for sol in best_n_solutions:
        list_of_occurence += sol.nodes
    probability_vector = dict(Counter(list_of_occurence))
    probability_vector = {node: probability_vector[node]/best_n for node in probability_vector.keys()}
    
    # Stores probability of nodes not occuring in best_n solutions as small_value in probability_vector
    rest_of_nodes = [node for node in all_nodes if node not in probability_vector.keys()]
    probability_vector = probability_vector | {node: small_value for node in rest_of_nodes}
    
    return probability_vector

def update_prob_v(probability_vector, solution_list):
    best_n_solutions = sorted(solution_list,key=attrgetter('fitness'))[:best_n]
    list_of_occurence = []
    for sol in best_n_solutions:
        list_of_occurence += sol.nodes
    occurences = dict(Counter(list_of_occurence))
    
    # Update probability_vector based on current probability and probability of node in best_n solutions.
    # Weighting is determined by parameter lambda (lmbda). 
    # Lower lambda means more conservative updating while a higher lmbda value leads to more aggressive updating of the probability_vector
    for node in occurences:
        probability_vector[node] = (1-lmbda)*probability_vector[node] + lmbda*(occurences[node]/best_n)
    return probability_vector

def onlooker_bee_phase(graph, probability_vector, solution_list):
    seed()
    for bee in range(onlooker_bees):
        # Randomly choose a solution to explore further
        index = choice(range(len(solution_list)))
        new_solution = solution(graph, solution_list[index].mutate(probability_vector))
        new_solution.repair()
        new_solution.improve()
        if new_solution.fitness_score() < solution_list[index].fitness_score():
            solution_list[index] = deepcopy(new_solution)
        best_iteration = best(solution_list)
        return best_iteration, solution_list
        
def get_mds():
    # Calculates minimum weight dominating set for each graph in parallel and stores the solutions in mds_dict
    mds_dict = {graph: abc_eda.remote(graphs[graph]) for graph in graphs}
    mds_dict = {graph: ray.get(mds_dict[graph]) for graph in graphs}
    return mds_dict        

###### Output Processing #####

def get_graphs(directory):
    graphs = {}
    for file in directory.iterdir(): #iterate over files in directory
        if str(file).endswith(".csv"):
            graph_name = file.stem #set name of graph as filename without extension
            edge_list = pd.read_csv(file) #import edge list
            edge_list = edge_list[edge_list['Parm1'] != edge_list['Parm2']]
            G = nx.from_pandas_edgelist(edge_list, source = 'Parm1', target = 'Parm2', edge_attr = 'weight')
            graphs[graph_name] = G
    return graphs

def get_doublets():
    double_list = {}
    for graph in graphs:
        double_list[graph] = []
        for component in nx.connected_components(graphs[graph]):
            if len(component) == 2:
                double_list[graph].append(component)
    return double_list

def remove_doublets(mds):
    for graph in graphs:
        for component in nx.connected_components(graphs[graph]):
            if len(component) == 2:
                for node in component:
                    if node in mds[graph]:
                        mds[graph].remove(node)

def get_isolated_nodes():
    isol_nodes = {}
    for i in graphs:
        isol_nodes[i] = list(nx.isolates(graphs[i]))
    return isol_nodes

def analyze(ds, isol_nodes, double_list):
    Path(output_dir).mkdir(parents = True, exist_ok = True, mode = 0o666)
    file_path = output_dir.as_posix() + '/{path}_ds.csv'.format(path = input_dir.name)
    with open(file_path, 'w') as file:
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

