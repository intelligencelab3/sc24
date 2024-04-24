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
