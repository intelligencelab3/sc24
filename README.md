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
 # 1. Data generation 
 ```
scripts is in dataset_prep/dataset_generator.py.
class ABCGenDataset:
```
gentype =0 CSA-array Multiplier generation and labeling
gentype =1 CPA Adder generation and labeling
gentype =2 Read a design and generate dataset
gentype =3 Generate Booth-encoded multiplier (tbd)
