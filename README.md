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
