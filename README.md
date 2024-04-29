# GROOT: Graph Edge Re-growth and Partitioning for the Verification of Large Designs in Logic Synthesis

## Installation

### Prerequisites
- Python packages: `torch` and `torch_geometric`
- Readline package (follow ABC installation requirements):
  - Ubuntu: `sudo apt-get install libreadline6 libreadline6-dev`
  - CentOS/RedHat: `sudo yum install readline-devel`

### Compile ABC Customized for Graph Learning
```bash
cd abc
make clean
make -j4
```

### Implementation
 #### 1. Data generation 
 ```
scripts is in dataset_prep/dataset_generator.py.
class ABCGenDataset:
```
gentype =0 CSA-array Multiplier generation and labeling

gentype =1 CPA Adder generation and labeling

gentype =2 Read a design and generate dataset

gentype =3 Generate Booth-encoded multiplier (tbd)

Note: ABC is required (../abc) (make sure to create a link symbol of abc binary in this folder)

`ln -s ../abc/abc .`

ABC implementation for data generation.
```
// abc/src/proof/acec/acecXor.c
class Gia_EdgelistMultiLabel()
```
#### 2.Train-Test Demo - training on 8-bit CSA and predicting on 32-bit CSA
#### Dataset Generation
```bash
python ABC_dataset_generation.py --bits 8
# generate an 8-bit CSA multiplier
```
```

python ABC_dataset_generation.py --bits 32
# generate a 32-bit CSA multiplier
```
##### Training and Inference

``` bash
python gnn_multitask.py --bits 8 --bits_test 32
# training with mult8, and testing with mult32
```

##### Inference with pre-trained model
```bash
python gnn_multitask_inference.py --model_path SAGE_mult8 --bits_test 32 --design_copies 1
# load the pre-trained model "SAGE_mult8", and test with mult32
```

Training INPUT: 8-bit CSA-Mult

Testing INPUT: 32-bit CSA-Mult

```
# training
Highest Train: 99.45
Highest Valid: 100.00
  Final Train: 98.90
   Final Test: 99.12

# testing
mult32
Highest Train: 0.00 ± nan
Highest Valid: 0.00 ± nan
  Final Train: 0.00 ± nan
   Final Test: 99.95 ± nan
```
New commands for ABC
```

abc 01> edgelist -h
# usage: edgelist: Generate pre-dataset for graph learning (MPNN, GraphSAGE, dense graph matrix)
# -F : Edgelist file name (*.el)
# -c : Class map for corresponding edgelist (Only for GraphSAGE; must have -F -c -f all enabled)
# -f : Features of nodes (Only for GraphSAGE; must have -F -c -f all enabled)
# -L : Switch to logic netlist without labels (such as AIG and LUT-netlist)

```


##### Run the Groot-GPU (modified Accel-GCN)

This includes a python script block_level_partition.py that preprocess the graph datasets further into the needed metadata, from main.cu to util.h that are run as CUDA testbed for our modified Accel-GCN, namely Groot-GPU.


### Prerequisites
Nvidia GPU with compute capability greater than or equal to 8.6

CUDA toolkit 12.0

cmake version 3.5

For the python scripts, numpy and scipy are required


### Download dataset
Our benchmark dataset contains 18 graphs:
![benchmark graphs](images/18graphs.png)

It can be downloaded from https://drive.google.com/file/d/1_sE65oveGpzRdCcExBmUaNG982lUB-Cx/view?usp=drive_link , 
or you can use the following command:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_sE65oveGpzRdCcExBmUaNG982lUB-Cx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_sE65oveGpzRdCcExBmUaNG982lUB-Cx" -O 18graphs.tar.gz && rm -rf /tmp/cookies.txt
```
Place the downloaded file in the project directory, then unzip it (and rename it).
```
tar xzvf 18graphs.tar.gz
mv 18graphs graphs
```
### Other datasets

Based on our EDA tasks, we prepare graph datasets from Booth, Technology, and Mapped datasets, where you can:
```
tar xzvf graphs_{name of dataset}.tar.gz
mv graphs_{name of dataset} graphs
```

Generate block-level partitioning meta-data.
```
mkdir block_level_meta
python block_level_partition.py
```

### Compilation
```
mkdir build
cd build
cmake ..
make -j10
```

## Benchmarking
Benchmark SPMM kernels on a specified graph and a specified right-hand matrix column dimension:
```
./spmm_test {graph's name} 60
```
If no parameters are attached, 
it will execute a traversal-style benchmark for all graphs and all right-multiply matrix column dimensions 
(controlled by `dim_min`, `dim_max`, and `interval` in `main.cu`):
```
./spmm_test
```
You can use a pipe to save the results: 
```
./spmm_test > result.txt
```
