# Introduction 

The code is attached to the paper ``Improved Rate of First Order Algorithms for Entropic Optimal Transport'', Yiling Luo, Yiling Xie, Xiaoming Huo, AISTATS 2023.
To reproduce all the experimental results in the paper, download the `mirror_experiment` filefolder and run the `run_experiment.ipynb` file. 


# Code files

`mirror_experiment/utils.py`: Toolbox for calculating gradient, converting dual variable to primal variable, rounding primal solution to feasible region, simulating images, and computing distance for image marginal distributions.

`mirror_experiment/algos.py`: Includes the implementation of PDASMD(-B) algorithm and all other algorithms that we compare in the experiment. Some implementations refer to https://github.com/PythonOT/POT/tree/master/ot, https://github.com/JasonAltschuler/OptimalTransportNIPS17 and https://github.com/nazya/AAM. 
