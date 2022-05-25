# Multi-layer State Evolution Under Random Convolutional Design

_This code is provided as anonymous supplemental material during the NeurIPS 2022 review process. Please do not distribute_. 

---

## Installation

To install the packages required to run this code, use the command `conda create --name <env> --file Requirements.txt`.


## Running the Code 
The code can be run using command line argument specied in `main.py`. See the documentation by running `python main.py`. 

Here are some examples:

- For Figure 1, sparsity prior, with varying `a` (measurement ratio) and `sparsity` (average number of nonzero entries of prior samples):

`python main.py --name Fig-1 --prior sparse --channel conv-cs --noise_std 0.01 -a {a} --sparsity {sparsity} --dims 1024 1024 3`

- For Figure 5, ReLU prior, with varying `a` (measurement ratio) and `L` (number of layers of the prior): 

`python main.py --name Fig-5-ReLU --prior {L}-relu --channel cs --noise_std -0.01 -a {a} --dims 10000 10 3`

- For Figure 5, Linear prior, with varying `a` (measurement ratio) and `L` (number of layers of the prior): :

`python main.py --name Fig-5-Linear --prior {L}-relu --channel cs --noise_std -0.01 -a {a} --dims 10000 10 3`

Note that including the `--se` flag will cause the code to run the state evolution simulation instead. 