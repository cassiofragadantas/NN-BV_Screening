This folder contains the original MATLAB implementation of the Active Set Newton Algorithm(ASNA) by Tuomas Virtanen

 

The algorithm is described in the following articles

T. Virtanen, J. F. Gemmeke, and B. Raj. Active-Set Newton
Algorithm for Overcomplete Non-Negative Representations of Audio,
IEEE Transactions on  Audio Speech and Language Processing,
volume 21 issue 11, 2013.

T. Virtanen, B. Raj, J. F. Gemmeke, and H. Van hamme, Active-set
Newton algorithm for non-negative sparse coding of audio, in proc.
International Conference on Acoustics, Speech and Signal
Processing, 2014, submitted for publication 

See http://www.cs.tut.fi/~tuomasv/software.html for more information.



Citation
---------
When used cite the articles mentioned in the previous section.


USAGE
------------

The file containing the main function of the algorithm is asna.m

ASNA (Active-Set Newton Algorithm) minimizes the Kullback-Leibler
divergence between non-negative observations X and the model A*S,
plus L1-norm of A weighted by lambda, by estimating non-negative
weights A.

    usage: A=asna(X,S,lambda,iterations)

    output 

    A          activations (observations x atoms)
    q          quality metrics

    input:

    X           non-negative observation matrix (observations x frequencies)
    S           non-negative dictionary matrix (atoms x frequencies)
    lambda      sparseness costs (scalar, or frames x atoms, or frames
             x 1, or 1 x atoms)
    iterations  maximum number of iterations (default = 500)

