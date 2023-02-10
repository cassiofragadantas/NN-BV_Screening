Copyright (C) 2017 by Pablo San Juan Sebastian                        
p.sanjuan@upv.es                                                      
                                                                      
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.                      

Citation
---------

When using this code you should  cite the following papers:

P. San Juan, T. Virtanen, V. M. Garcia-Molla, A. M. VIdal
Efficient Parallel Implementation of Active-Set Newton Algorithm for Non-Negative Sparse Representations,
Computational and Mathematical Methods in Science and Engineering CMMSE 2017

T. Virtanen, J. F. Gemmeke, and B. Raj. Active-Set Newton
Algorithm for Overcomplete Non-Negative Representations of Audio,
IEEE Transactions on  Audio Speech and Language Processing,
volume 21 issue 11, 2013.

Prerequisites
--------------


This code needs an implementation of BLAS/LAPACK installed in the sistem.

This code has been tested with Intel Math Kernel Library(https://software.intel.com/en-us/articles/intel-math-kernel-library-documentation) and OpenBLAS (http://www.openblas.net/)

When using intel  MKL the compiler flag -DWith_MKL should be added.

When using the parallel version the OpenMP threading library (http://www.openmp.org/) is needed.


Compilation examples
---------------------
Using intel compiler and MKL:

icc -O3 -DWith_MKL -DWith_ICC asna.c asna.h mainAsna.c -mkl -qopenmp -o mainAsna

Using gcc and OpenBLAS library:

gcc -O3 asna.h asna.c mainAsna.c -fopenmp -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib/ -lm -lopenblas -lgfortran -o mainAsnaGcc


Execution example
---------------------

The explample main provided reads the original matrix X and the dictionary matrix B from files and executes the algorithm setting iter as maximum number opf iterations.
The repetition parameter is meant for timing purposes, it repeats the algorithm as many times as indicated and then averages the time.


    Apply Asna algorithm. X and S are double precision matrices loaded from file.
        X: is an f x o matrix.
        B: is an f x n matrix.
        iter: number of iterations.
        iter: number of repetitions.
    Usage: ./mainAsna X B <iter> <repe>


    ./mainAsna matrix.dat dictionary.dat 500 1
    Entering asna
    #  M    N     K      Iteration    Time (sec)      Error
    # --------------------------------------------------------------------
    751   130  10000     500        1.92632E+00    6.02586384E+03
 
> The file structure used in the main example is 2 integers giving the size of the matrix followed by the matrix in double precision format and colum major order. 
>Written in binary mode.


 
