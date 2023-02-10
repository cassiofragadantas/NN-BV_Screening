Improved version of the original MATLAB Active Set Netwon ALgorithm(ASNA) by Pablo San Juan

The improvements done to the original code are explained in the following paper:

P. San Juan, T. Virtanen, V. M. Garcia-Molla, A. M. VIdal
Efficient Parallel Implementation of Active-Set Newton Algorithm for Non-Negative Sparse Representations,
Computational and Mathematical Methods in Science and Engineering CMMSE 2017

Citations
-----------

When used cite the article mentioned in the previous section.


Usage
----------

The file containing the main function of the algorithm is asnaRework.m

    usage: A=asnaRework(X,S,lambda,iterations)

    output 

    A          activations (atoms x observations)
    
    input:

    X           non-negative observation matrix (frequencies x observations)
    S           non-negative dictionary matrix (frequencies x atoms)
    lambda      sparseness costs (scalar, or frames x atoms, or frames
                 x 1, or 1 x atoms)
    iterations  maximum number of iterations (default = 500)