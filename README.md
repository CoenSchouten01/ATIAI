## Welcome to the repository for the project of group 1 (Coen Schouten and Kelvin Toonen) of TU/e course 2AMM40 Advanced Topics in Artificial Intelligence

In this repository you can find the code used for the experiments in our report: Increasing the complexity of "SGD on neural networks learns functions of increasing complexity". Furthermore one can find the plots with all of the experimental results used for the report in the folder "Experimental Results".

### Using the code
Start off by making sure the requirements are satisfied by running pip install -r requirements.txt in the terminal

Next configurate main.py in such a way that it runs the experiments you are interested in, for this one can change the following variables:

global variables:
- DATASET: The dataset on which to run the experiments, either "MNIST" or "CIFAR10"
- MULTICLASS: wether or not to run the multiclass extension, boolean value.

variables within main loop:
- n_experiments: number of repeated experiment runs
- epochs: number of training epochs, recommended value is 50 for MNIST and 100 for CIFAR10

fit parameters:
- model_1: The "complex" model whose performance we want to explain by the "simple" model
- optimizer_1: The optimizer corresponding to the "complex" model
- model_2: The "simple" conditioning model used to explain the "complex" model
- optimizer_2: The optimizer corresponding to the "complex" model

Then simply run python main.py to execute the experiments.