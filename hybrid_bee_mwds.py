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
from dataclasses import dataclass

import networkx as nx
import ray
import pandas as pd

import parameters


def main():
    start_time = time.perf_counter()

    # Input
    input_dir = Path(os.path.abspath(sys.argv[1]))
    output_dir = Path(os.path.abspath(sys.argv[2]))

    # Store global parameters
    global_parameters = stored_parameters(
        input_dir,
        output_dir,
        parameters.population_size,
        parameters.max_trials,
        parameters.max_iterations-1,
        parameters.onlooker_bees,
        parameters.d,
        parameters.beta,
        lmbda=parameters.lmbda,
        mu1=parameters.mu1,
        mu2=parameters.mu2,
        small_value=parameters.small_value
    )



    # Calculate MWDS in parallel
    ray.init(num_cpus=int(sys.argv[3]))
    mwds = dominating_set(get_mds(global_parameters), global_parameters)
    ray.shutdown()

    # Remove doublets from mds
    mwds.remove_doublets()

    # Save csv with graphs and corresponding mwds, doublets and isolated nodes
    mwds.export()

    total_time = time.perf_counter() - start_time
    print("Finished in {}".format(total_time))


@dataclass
class stored_parameters:
    # Input
    input_dir: Path
    output_dir: Path

    # Integer parameters
    population_size: int
    max_trials: int
    max_iterations: int
    onlooker_bees: int
    d: int

    # Probability parameters
    beta: float
    lmbda: float
    mu1: float
    mu2: float
    small_value: float

    def get_graphs(self):
        graphs = {}
        isolates = {}
        for file in self.input_dir.iterdir():  # iterate over files in directory
            if str(file).endswith(".csv"):
                graph_name = file.stem  # set name of graph as filename without extension
                edge_list = pd.read_csv(file)  # import edge list
                G = nx.from_pandas_edgelist(
                    edge_list, source='Parm1', target='Parm2', edge_attr='weight')
                G.remove_edges_from(nx.selfloop_edges(G))
                isolates[graph_name] = list(nx.isolates(G))
                G.remove_nodes_from(isolates[graph_name])
                graphs[graph_name] = G
        return graphs, isolates

    def __post_init__(self):
        self.graphs, self.isolates = self.get_graphs()
        self.best_n = int(round(self.population_size/self.d))

class solution_list:
    def __init__(self, graph, global_parameters):
        self.parameters = global_parameters
        self.graph = graph
        self.solutions = self.init_pop()
        self.best_solution = self.best()
        self.best_iteration = deepcopy(self.best_solution)
        self.probability_vector = self.init_prob_v()

    def init_pop(self):
        # Generates initial population of solutions
        # First solution is an empty repaired solution
        # Second solution is solution containing all nodes that has been improved
        # The rest of the solutions are generated by adding random nodes until valid
        first_solution = solution(self.graph, self.parameters)
        first_solution.repair()
        second_solution = solution(
            self.graph, self.parameters, list(self.graph.nodes()))
        second_solution.improve()
        rest = [solution(self.graph, self.parameters)
                for i in range(2, self.parameters.population_size)]
        for sol in rest:
            sol.random_solution()
        solutions = [first_solution, second_solution] + rest
        return solutions

    def best(self):
        for solution in self.solutions:
            solution.update_fitness()
        best_solution = min(self.solutions, key=attrgetter('fitness'))
        return best_solution

    def init_prob_v(self):
        all_nodes = list(self.graph.nodes())
        best_n_solutions = sorted(self.solutions, key=attrgetter('fitness'))[
            :self.parameters.best_n]

        # Convert frequency of node in best_n solutions to probability and store it probability_vector
        list_of_occurence = []
        for sol in best_n_solutions:
            list_of_occurence += sol.nodes
        probability_vector = dict(Counter(list_of_occurence))
        probability_vector = {
            node: probability_vector[node]/parameters.best_n for node in probability_vector.keys()}

        # Stores probability of nodes not occuring in best_n solutions as small_value in probability_vector
        rest_of_nodes = [
            node for node in all_nodes if node not in probability_vector.keys()]
        probability_vector = probability_vector | {
            node: parameters.small_value for node in rest_of_nodes}

        return probability_vector

    def update_prob_v(self):
        best_n_solutions = sorted(self.solutions, key=attrgetter('fitness'))[
            :parameters.best_n]
        list_of_occurence = []
        for sol in best_n_solutions:
            list_of_occurence += sol.nodes
        occurences = dict(Counter(list_of_occurence))

        # Update probability_vector based on current probability and probability of node in best_n solutions.
        # Weighting is determined by parameter lambda (lmbda).
        # Lower lambda means more conservative updating while a higher lmbda value leads to more aggressive updating of the probability_vector
        for node in occurences:
            self.probability_vector[node] = (
                1-parameters.lmbda)*self.probability_vector[node] + parameters.lmbda*(occurences[node]/parameters.best_n)

    def onlooker_bee_phase(self):
        seed()
        for bee in range(parameters.onlooker_bees):
            # Randomly choose a solution to explore further
            index = choice(range(len(self.solutions)))
            new_solution = solution(
                self.graph, self.solutions[index].mutate(self.probability_vector))
            new_solution.repair()
            new_solution.improve()
            if new_solution.fitness_score() < self.solutions[index].fitness_score():
                self.solutions[index] = deepcopy(new_solution)
            self.best_iteration = self.best()



class solution:
    def __init__(self, graph, global_parameters, nodes=set()):
        self.parameters = global_parameters
        self.graph = graph
        self.edge_list = nx.to_pandas_edgelist(
            self.graph, source='Parm1', target='Parm2')
        self.nodes = set(nodes)
        self.fitness = self.fitness_score()  # lower = better

    def random_solution(self):
        seed()
        self.nodes = set()
        while nx.is_dominating_set(self.graph, self.nodes) == False:
            self.nodes.add(choice(list(self.graph.nodes())))
        self.update_fitness()

    def fitness_score(self):
        # sum of weights associated with nodes in solution
        fitness_score = self.edge_list.loc[self.edge_list['Parm2'].isin(list(
            self.nodes)) + self.edge_list['Parm1'].isin(list(self.nodes)), 'weight'].sum()
        # add penalty per number of nodes
        fitness_score += len(self.nodes)*0.01
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

        not_in_solution = [node for node in list(
            self.graph.nodes()) if node not in self.nodes]

        # Store node values in dictionary
        vertex = {}
        for node in not_in_solution:
            vertex[node] = {}
            undominated_neighbors = [node for node in list(
                self.graph.neighbors(node)) if node not in dominated]
            vertex[node]['weight_sum'] = self.graph.degree(node, 'weight')
            vertex[node]['degree'] = self.graph.degree(node)
            vertex[node]['weight_sum_of_non_dominated_neighbors'] = 0
            for neighbor in undominated_neighbors:
                vertex[node]['weight_sum_of_non_dominated_neighbors'] += self.graph.degree(
                    neighbor, 'weight')

        # Calculate best candidate to include in solution in three different ways
        heuristics = {1: {}, 2: {}, 3: {}}
        for node in vertex:
            heuristics[1][node] = vertex[node]['degree'] / \
                vertex[node]['weight_sum']
            heuristics[2][node] = vertex[node]['weight_sum_of_non_dominated_neighbors'] / \
                vertex[node]['weight_sum']
            heuristics[3][node] = (vertex[node]['weight_sum_of_non_dominated_neighbors']
                                   * vertex[node]['degree'])/vertex[node]['weight_sum']

        # Randomly selects one of three calculations to use
        random_choice = choice([1, 2, 3])
        best_node = max(heuristics[random_choice],
                        key=heuristics[random_choice].get)
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
        heuristics = {node: vertex[node]['weight_sum'] /
                      vertex[node]['degree'] for node in vertex.keys()}
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
            if random() < parameters.beta and random() < probability_vector[node]:
                mutation.add(node)
            elif node in self.nodes:
                mutation.add(node)
        return mutation

    def emp_bee_phase(self, probability_vector):
        # Mutates current solution until no improvement has been made for max_trials
        trials_with_no_change = 0
        while trials_with_no_change <= parameters.max_trials:
            mutation = solution(self.graph, self.mutate(probability_vector))
            if mutation.fitness_score() < self.fitness_score():
                self = deepcopy(mutation)
            else:
                trials_with_no_change += 1

    def scout_bee_phase(self, best_iteration):
        seed()
        # After employed_bee_phase() exhausted local solution space:
        # mu1 is the probability of selecting a new completely random solution
        if random() < parameters.mu1:
            self.random_solution()
        # mu2 is the probability of each node in the best solution being used to construct the new solution instead
        else:
            self.nodes = set()
            for node in best_iteration.nodes:
                if random() < parameters.mu2:
                    self.nodes.add(node)
            self.repair()


class dominating_set:
    def __init__(self, ds, global_parameters):
        self.ds = deepcopy(ds)
        self.parameters = deepcopy(global_parameters)
        self.doublets = self.get_doublets()
        self.isolated_nodes = self.parameters.isolates

    def get_doublets(self):
        double_list = {}
        for graph in self.parameters.graphs:
            double_list[graph] = []
            for component in nx.connected_components(self.parameters.graphs[graph]):
                if len(component) == 2:
                    double_list[graph].append(component)
        return double_list

    def remove_doublets(self):
        for graph in self.parameters.graphs:
            for component in nx.connected_components(self.parameters.graphs[graph]):
                if len(component) == 2:
                    for node in component:
                        if node in self.ds[graph]:
                            self.ds[graph].remove(node)

    def get_isolated_nodes(self):
        isol_nodes = {}
        for i in self.parameters.graphs:
            isol_nodes[i] = list(nx.isolates(self.parameters.graphs[i]))
        return isol_nodes

    def export(self):
        self.parameters.output_dir.mkdir(parents=True, exist_ok=True)
        self.parameters.output_dir.chmod(0o776)
        file_path = "{}/{}/{}_ds.csv".format(self.parameters.output_dir.as_posix(), 
                                          self.parameters.input_dir.name,
                                          self.parameters.input_dir.name)
        with open(file_path, 'w') as file:
            file.write("subset_threshold,dominating_set,doublets,isolated_nodes\n")
            for graph in self.ds.keys():
                ds_list = str(self.ds[graph])
                ds_list = '"{}"'.format(ds_list[1: len(ds_list) - 1])
                doubles_list = str(self.doublets[graph])
                doubles_list = '"{}"'.format(
                    doubles_list[1: len(doubles_list) - 1])
                isol_list = str(self.isolated_nodes[graph])
                isol_list = '"{}"'.format(isol_list[1: len(isol_list) - 1])
                file.write('{},{},{},{}\n'.format(graph, ds_list, doubles_list, isol_list))
        
        file_path = Path("{}/{}/logs/".format(
            self.parameters.output_dir.as_posix(),
            self.parameters.input_dir.name))
        logs = {}
        for file in file_path.iterdir():
            with open(file, 'r') as f:
                logs[file.stem] = f.read()
            file.unlink()
        Path.rmdir(file_path)
        file_path = Path("{}/{}/log.csv".format(
            self.parameters.output_dir.as_posix(),
            self.parameters.input_dir.name))
        with open(file_path,'w') as f:
            f.write('graph,iterations\n')
            for key, iterations in logs.items():
                f.write('{},{}\n'.format(key, iterations))
                

@ray.remote
def abc_eda(key, graph, global_parameters):
    seed()
    # Initialize starting population of solutions and probability_vector
    hive = solution_list(graph, global_parameters)

    # Explore solutions while updating probability_vector after each iteration
    iterations = 0
    log_iteration = 1
    local_iterations = global_parameters.max_iterations
    while iterations <= local_iterations:
        iterations += 1
        for bee in hive.solutions:
            bee.emp_bee_phase(hive.probability_vector)
            hive.best_iteration = hive.best()
            bee.scout_bee_phase(hive.best_iteration)
        hive.onlooker_bee_phase()
        hive.update_prob_v()
        if hive.best_iteration.fitness_score() < hive.best_solution.fitness_score():
            log_iteration = iterations
            hive.best_solution = deepcopy(hive.best_iteration)
            print('{}: New Best solution found after {} iterations'.format(key, iterations))
            if iterations > local_iterations/2:
                local_iterations += 1
    print('Max ({}) iterations reached for {}'.format(iterations, key))
    
    logdir_path = Path("{}/{}/logs/".format(
        global_parameters.output_dir.as_posix(),
        global_parameters.input_dir.name))
    logdir_path.mkdir(parents=True, exist_ok=True)
    file_path = "{}/{}/logs/{}.log".format(
        global_parameters.output_dir.as_posix(),
        global_parameters.input_dir.name,
        key)
    with open(file_path,'w') as file:
        file.write(str(log_iteration))
    return list(hive.best_solution.nodes)


def get_mds(global_parameters):
    # Calculates minimum weight dominating set for each graph in parallel and stores the solutions in mds_dict
    print('Global Parameters: \n{}'.format(global_parameters))
    print("Calculating MWDS for following graphs:")
    for graph in sorted(list(global_parameters.graphs)):
        print(graph)
    mds_dict = {}
    for key, graph in global_parameters.graphs.items():
        mds_dict[key] = abc_eda.remote(key, graph, global_parameters)

    mds_dict = {graph: ray.get(mds_dict[graph])
                for graph in global_parameters.graphs}
    return mds_dict

if __name__ == '__main__':
    main()