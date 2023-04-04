# NN-BV_Screening
Safe screening for non-negative and bounded-variable linear regression

# Instructions to reproduce the paper results
## Bounded-variable LS experiments
Run the script `exp/Exp_BV.m`. 

- Experiment by varying some parameters:
    - **Solver**: set `solver = 'PD'` to run the primal-dual solver (Prox. Grad. solver `solver = 'PG' being the default).
    - **Dimensions**: freely change the problem dimensions (n, m).
    - **Dataset**: to run the experiments with the hyperspectral data (Figure 4) set `exp_type = 'Cuprite_USGS-lib'`. You will need to download the Cuprite image data [here](http://www.lx.it.pt/~bioucas/code/cuprite_ref.zip) and place the mat file it in the folder `./datasets`. For the exact same setup as in the paper, also set `sigma=0.1`.
    - **Oracle dual point**: can be desactivated by setting `oracle_dual = false`.

Run the script `exp/Exp_BV_boxval.m` to reproduce Figure 1 (speedup vs. saturation ratio).

## Non-negative LS experiments
Run the script `exp/Exp_NN.m`. 

- You can also experiment by varying some parameters:
	- **Solver**: activate solvers by setting to `true` the variables CoD, ActiveSet or MM (slow).
	- **Dimensions**: freely change the problem dimensions (n, m)
	- **Dataset**: to run the archetypal analysis experiment (Figure 5) set `exp_type = 'NIPSpapers'`. You will need to download the Cuprite image data [here](http://ai.stanford.edu/~gal/Data/NIPS/nips_1-17.mat) and place it in the folder `./datasets`.


# Codebase descritption

The project can be divided into the following parts and corresponding folders:
1. **Screening**: Safe screening functions, placed at the root folder `./`
2. **Solvers**: variants with and without screening, placed at the folder `solvers/`
3. **Experiments**: main scripts for generating experimental comparisons, placed at `exp/`
4. **General utils**: General utility files, be them scripts or data, placed at `utils/` 

## 1. Screening
Files concerning this part:
- `bvGapSafeScreen.m`: implements screening for the bouded-variable least-squares (BVLS) problem.
- `nnGapSafeScreen.m` implements screening for the non-negative least-squares (BVLS) problem.

## 2. Solvers
Files concerning this part:
- [TO COMPLETE]

## 3. Experiments
Files concerning this part:
- `Exp_BV.m`: BVLS experiments (see above)
- `Exp_BV_boxval.m`: BVLS experiments with varying box sizes.
- `Exp_NN.m`: NNLS experiments (see above)
- `Exp_NN_DualDirection.m`: experiments with different dual directions. Used to generate Figure 2 in the paper. The different directions need to be uncommented from the code (around line 40)

## 4. General utils
Files concerning this part:
- [TO COMPLETE]

