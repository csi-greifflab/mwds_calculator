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

## Algorithm Parameters
