# abc_eda

Algorithm to compute minimum weight dominating sets of undirected weighted networks. Adapted from Singh's and Shetgaonkar's publication, "Hybridization of Artificial Bee Colony Algorithm with Estimation of Distribution Algorithm for Minimum Weight Dominating Set Problem" published as part of the Advances in Intelligent Systems and Computing book series (AISC, volume 1270) in 2020.


## Install

To run the script clone the directory and install dependencies via pip:
```bash
git clone https://github.com/jahnzh/abc_eda/
cd abc_eda
pip install -r requirements.txt
```

## Usage

The script takes a directory with adjacency matrices in csv format as an input and another directory to store the output.
The csv file has 3 columns "Parm1", "Parm2" and "weight" where "Parm1" and "Parm2" correspond to vertices and "weight" to the weights of edges inside the network.

```bash
python hybrid_bee_mds.py <input_dir> <output_dir> <number_of_cpus>
```

### Algorithm Parameters

**Integer parameters:**
| Parameter (= default)  | Explanation  |
| ------------ | ------------ |
|  population_size = 50 |  Size of initial solution_list (width)|
|  max_trials = 2 |Depth of exploration of each individual solution   |
|  max_iterations = 30 |Number of iterations for entire algorithm   |
|  onlooker_bees = 20 |  Number of bees randomly choosing a solution to explore further |
| d = 5   | Number to determine best n solutions (best_n = population_size/d)  |


**Probability parameters:**
| Parameter (= default)  | Explanation  |
| ------------ | ------------ |
|   beta = 0.2 |  Aggresiveness of solution mutations|
|  lmbda = 0.6 |Aggresiveness of probability_vector updates |
|  mu1 = 0.5  |Probability of selecting a new completely random solution during scout bee phase |
|  mu2 = 0.5    | Probability of including each node from the best solution in a new solution during scout bee phase |
| small_value = 0.05   | Initial non-zero value for probability_vector  |
