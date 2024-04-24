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



### Implementation
# 1. Data Generator
File: dataset_prep/dataset_generator.py
Class: ABCGenDataset
Generation Types (gentype)
0: CSA-array Multiplier generation and labeling
1: CPA Adder generation and labeling
2: Read a design and generate dataset
3: Generate Booth-encoded multiplier (tbd)
Note: ABC is required (ensure a symbolic link to the ABC binary in the current folder):

bash
ln -s ../abc/abc .
