/***************************************************************************
 *   Copyright (C) 2017 by Pablo San Juan Sebastian                        *
 *   p.sanjuan@upv.es                                                      *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 * The algorithm on wich this implementation is based is described in      *
 * the following articles                                                  *
 * T. Virtanen, J. F. Gemmeke, and B. Raj. Active-Set Newton               *
 * Algorithm for Overcomplete Non-Negative Representations of Audio,       *
 * IEEE Transactions on  Audio Speech and Language Processing,             *
 * volume 21 issue 11, 2013.                                               *
 *                                                                         *
 * See https://P.SanJuan@gitlab.com/P.SanJuan/ASNA.git for latest version  *
 ***************************************************************************/


#include "asna.h"


/**
 *  \fn    void dmultOnes(const int f, const int n, const int o, const double* M, double* __restrict__ MO)
 *  \brief This function performs the operation MO = M * ones(f,o)
 *  \param f: (input) Number of features
 *  \param n: (input) Number of dictionary atoms
 *  \param o: (input) Number of observations
 *  \param M: (input) Double precision matrix
 *  \param MO: (output) Double precision matrix
*/
void dmultOnes(const int f, const int n, const int o, const double* M, double* __restrict__ MO)
{
	double* Ones;
	
	#ifdef With_MKL
        Ones = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
    #else
        Ones = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
    #endif
	
	dmemset_x86(f*o,Ones,1.0);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,o,f,1,M,n,Ones,f,0,MO,n);

	#ifdef With_MKL
        mkl_free(Ones);
    #else
        _mm_free(Ones);
    #endif 
}

/**
 *  \fn    void compInitialAtoms(const int f, const int n, const int o, const double* X, const double* B, int* __restrict__ initAtoms, double* __restrict__ initWeights)
 *  \brief This function performs the initialization os the ASNA algorithm comptuting 
 * 	the best weigth which minimizes alone the KL-divergence for each observation
 *  \param f: (input) Number of features
 *  \param n: (input) Number of dictionary atoms
 *  \param o: (input) Number of observations
 *  \param X: (input) Double precision matrix to factorize (f x o)(1D column-major)
 *  \param B: (input) Double precision dictionary matrix (f x n) (1D column-major)
 *  \param initAtoms: (output) Integer vector containing the index values of the initial atoms
 *  \param initAtoms: (output) Double precision vector containing the weigth for each initial atom
*/
void compInitialAtoms(const int f, const int n, const int o, const double* X, const double* B, int* __restrict__ initAtoms, double* __restrict__ initWeights)
{
	int i, j,k;
	double sX;
	double* sB, *Aux, *KLDiv, *W, *logB;
	
	#ifdef With_MKL
	    sB = (double *)mkl_malloc(n * sizeof(double), WRDLEN);
        Aux = (double *)mkl_malloc(o * sizeof(double), WRDLEN);
		W = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
		KLDiv = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
		logB = (double *)mkl_malloc(n * f * sizeof(double), WRDLEN);
    #else
        sB = (double *)_mm_malloc(n * sizeof(double), WRDLEN);
		Aux = (double *)_mm_malloc(o * sizeof(double), WRDLEN);
		W = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
		KLDiv = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
		logB = (double *)_mm_malloc(n * f * sizeof(double), WRDLEN);
    #endif
	
	//Compute sum(B)
	#pragma omp parallel for schedule(static)
	for(j = 0; j < n; j++)
		sB[j] = cblas_dasum(f,&B[j*f],1);
	
	//Compute W=bsxfun(@rdivide,sum(X,1), sum(B,1)');
	// KLDiv = bsxfun(@times,sum(X,1),log(A))
	#pragma omp parallel for schedule(static) private(sX,i,k)
	for(j = 0; j < o; j++)
	{
		sX = cblas_dasum(f,&X[j*f],1);
		for(i = 0; i < n; i++)
		{
			W[i + j*n] = sX / sB[i] ;
			KLDiv[i + j*n] = log(W[i + j*n]) * sX;
		}
		//Compute sum(X.*log(X));
		Aux[j] = 0;
		for(k = 0; k < f; k++)
		{
			Aux[j] = Aux[j] + X[k + j*f] * log(X[k + j*f]);
		}
		
	}
	
	//compute log(B)
	# pragma omp parallel for  schedule(static)
	for(i = 0; i < f*n; i++)
			logB[i] = log(B[i]);
	
	//Compute KLDiv = log(B)' * X + KLDiv
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,n,o,f,1,logB,f,X,f,1,KLDiv,n);
	
	//Compute KLDiv = KLDiv - expanded(Aux)
	#pragma omp parallel for  schedule(static) private(i)
	for(j = 0; j < o ; j++)
		for(i = 0; i < n; i++)
			KLDiv[i+ j *n] = KLDiv[i + j*n] -Aux[j];
	
	//Look for the minimum divergence for each observation (absooute)
	#pragma omp parallel for  schedule(static)
	for(j = 0; j < o; j++)
	{
		initAtoms[j] = cblas_idamin(n,&KLDiv[j*n],1);
		//initAtoms[j] = dMin_x86(n,&KLDiv[j*n]);
		initWeights[j] = W[initAtoms[j] + j * n];
	}
	

	#ifdef With_MKL
		mkl_free(sB);
        mkl_free(Aux);
		mkl_free(logB);
		mkl_free(KLDiv);
		mkl_free(W);
    #else
		_mm_free(sB);
        _mm_free(Aux);
		_mm_free(KLDiv);
		_mm_free(logB);
		_mm_free(W);
    #endif 
}


/**
 *  \fn   void addAtom(observation* obs, const int newIdx)
 *  \brief This function adds an atom with index newIdx to the last position of  the  active set of observation obs
 *  \param obs: (inout) Observation structure where the atom is going to be added
 *  \param rnewIdx: (input) Index to Add
*/
void addAtom(observation* obs, const int newIdx)
{
	int i,next,prev;
	atom* atomPtr=NULL;
	
	
	if(obs->nAtoms == 0)
	{
		#ifdef With_MKL
			obs->firstAtom= (atom*)mkl_malloc(sizeof(atom),WRDLEN);
		#else
			obs->firstAtom= (atom*)_mm_malloc(sizeof(atom),WRDLEN);
		#endif
		obs->lastAtom=obs->firstAtom;
		obs->nAtoms++;
		obs->firstAtom->atomIdx = newIdx;
		obs->firstAtom->prevAtom = NULL;
		obs->firstAtom->nextAtom = NULL;
	}
	else
	{
		#ifdef With_MKL
			atomPtr =(atom*)mkl_malloc(sizeof(atom),WRDLEN);
		#else
			atomPtr =(atom*)_mm_malloc(sizeof(atom),WRDLEN);
		#endif
		obs->lastAtom->nextAtom = atomPtr;
		atomPtr->atomIdx = newIdx;
		atomPtr->prevAtom = obs->lastAtom;
		atomPtr->nextAtom = NULL;
		obs->lastAtom = atomPtr;
		obs->nAtoms++;
	}
}


/**
 *  \fn   void removeAtom(observation* obs, const int remIdx)
 *  \brief This function removes an atom with index remIdx from the  active set of observation obs
 *  \param obs: (inout) Observation structure where the atom is going to be removed
 *  \param remIdx: (input) Index to remove
*/
void removeAtom(observation* obs, const int remIdx)
{
	int i;
	atom* atomPtr = NULL;
	
	for(i = 0, atomPtr = obs->firstAtom; i <  obs->nAtoms;i++)
	{
		if(atomPtr->atomIdx ==remIdx)
		{
			if(atomPtr->prevAtom != NULL)
				atomPtr->prevAtom->nextAtom = atomPtr->nextAtom;
			else
				obs->firstAtom = atomPtr->nextAtom;
			if(atomPtr->nextAtom != NULL)
				atomPtr->nextAtom->prevAtom = atomPtr->prevAtom;
			else
				obs->lastAtom = atomPtr->prevAtom; 
			obs->nAtoms--;
			
			#ifdef With_MKL
				mkl_free(atomPtr);
			#else
				_mm_free(atomPtr);
			#endif
			break;
		}
		
		if( i != obs->nAtoms -1)
			atomPtr = atomPtr->nextAtom;
	}
}

void freeObservations(int nObs,observation *obs)
{
	int i,j;
	atom* atomPtr;
	
	//atomPtr=obs[5].firstAtom;
	//mkl_free(atomPtr);
	
	for(i = 0; i < nObs; i++)
	{
		for(j=1; j< obs[i].nAtoms;j++)
		{
			atomPtr = obs[i].lastAtom;
			if(atomPtr == NULL)
				printf("Atomptr is null\n");
			obs[i].lastAtom = atomPtr->prevAtom;

			#ifdef With_MKL
			  mkl_free(atomPtr);
			#else
			  _mm_free(atomPtr)
			#endif
		}
		atomPtr = obs[i].lastAtom;
		
		#ifdef With_MKL
			mkl_free(atomPtr);
		#else
			_mm_free(atomPtr)
		#endif
	}
	
}

/**
 *  \fn    derrorAsna(const int m, const int n, const int k, const double *A, const double *W, const double *H)
 *  \brief This function returns double precision error when error is computed using kulback liebler divergence
 *  \param m:    (input) Number of rows of A and W
 *  \param n:    (input) Number of columns of A and H
 *  \param k:    (input) Number of columns/rows of W/H  
 *  \param A:    (input) Double precision input matrix A
 *  \param W:    (input) Double precision input matrix W
 *  \param H:    (input) Double precision input matrix H
*/
double derrorAsna(const int m, const int n, const int k, const double *A, const double *W, const double *H)
{
  double
    error=0.0,
    *tmp=NULL;
	
  #ifdef With_MKL
    tmp = (double *)mkl_malloc(m*n*sizeof(double), WRDLEN);
  #else
    tmp = (double *)_mm_malloc(m*n*sizeof(double), WRDLEN);
  #endif

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, W, m, H, k, 0.0, tmp, m);
  derrorbd1_x86(m*n, A, tmp);
  
  error=cblas_dasum(m*n, tmp, 1);
    
  #ifdef With_MKL
    mkl_free(tmp);
  #else
    _mm_free(tmp);
  #endif

  return error;
}


/**
 *  \fn   int dasna_cpu(const int f, const int n, const int o, const double* Xini, const double* Bini, double* W ,const int iter)
 *  \brief This function performs a matrix nonnegative approximation using Active-Set Newton Algorithm (ASNA) X = BW
 *  \param f: (input) Number of features
 *  \param n: (input) Number of dictionary atoms
 *  \param o: (input) Number of observations
 *  \param Xini: (input) Double precision matrix to factorize (f x o)(1D column-major)
 *  \param Bini: (input) Double precision dictionary matrix (f x n) (1D column-major)
 *  \param W: (output) Double precision weigths matrix (n x o) (1D column-major)
 *  \param iter: (input)Maximum number of iterations to perform
 *  \param nnz: (input) Maximum number of active atoms per observation (to restrict memory size of the hessian matrix, maximum n)
*/
int dasna_cpu(const int f, const int n, const int o, const double* Xini, const double* Bini, double* W ,const int iter, const int nnz)
{
  
	int i,j,k,k2,it, info, gPtr;
	int gradientComputed, minIdx, converged, nObs;
	int *initAtoms;
	double stepSize, num,norm;
    double *B, *X, *Bt, *BtO,*V, *R, *gradAll, *gradAll2, *Hwa;
    double *normsX, *normsB, *initWeights, *Mult, *Aux, *grad, *maxStep;
    observation* activeObs, *obsPtr;
	atom* atomPtr, *atomPtr2;
	
	#ifdef With_MKL
        X = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
		B = (double *)mkl_malloc(f * n * sizeof(double), WRDLEN);
		Bt = (double *)mkl_malloc(n * f * sizeof(double), WRDLEN);
		BtO = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
		V = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
		R = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
		gradAll = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
		gradAll2 = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
		Hwa = (double *)mkl_malloc(nnz * nnz * sizeof(double), WRDLEN);
		normsX = (double *)mkl_malloc(o * sizeof(double), WRDLEN);
		normsB = (double *)mkl_malloc(n * sizeof(double), WRDLEN);
		initAtoms = (int *)mkl_malloc(o * sizeof(int), WRDLEN);
		initWeights = (double *)mkl_malloc(o * sizeof(double), WRDLEN);
		Mult = (double *)mkl_malloc(f * sizeof(double), WRDLEN);
		Aux = (double *)mkl_malloc(f * sizeof(double), WRDLEN);
		grad = (double *)mkl_malloc(n * sizeof(double), WRDLEN);
		maxStep = (double *)mkl_malloc(n * sizeof(double), WRDLEN);
		activeObs = (observation*)mkl_malloc(o*sizeof(observation),WRDLEN);
    #else
        X = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
		B = (double *)_mm_malloc(f * n * sizeof(double), WRDLEN);
		Bt = (double *)_mm_malloc(n * f * sizeof(double), WRDLEN);
		BtO = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
		V = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
		R = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
		gradAll = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
		gradAll2 = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
		Hwa = (double *)_mm_malloc(nnz * nnz * sizeof(double), WRDLEN);
		normsX = (double *)_mm_malloc(o * sizeof(double), WRDLEN);
		normsB = (double *)_mm_malloc(n * sizeof(double), WRDLEN);
		initAtoms = (int *)_mm_malloc(o * sizeof(int), WRDLEN);
		initWeights = (double *)_mm_malloc(o * sizeof(double), WRDLEN);
		Mult = (double *)_mm_malloc(f * sizeof(double), WRDLEN);
		Aux = (double *)_mm_malloc(f * sizeof(double), WRDLEN);
		grad = (double *)_mm_malloc(n * sizeof(double), WRDLEN);
		maxStep = (double *)_mm_malloc(n * sizeof(double), WRDLEN);
		activeObs = (observation *)_mm_malloc(o * sizeof(observation), WRDLEN);
    #endif
	
	//init observation structure
	nObs = o;
	for(i = 0; i < o  ; i++)
	{
		activeObs[i].obsIdx=i;
		activeObs[i].converged=0;
		activeObs[i].nAtoms=0;
		activeObs[i].firstAtom=NULL;
		activeObs[i].lastAtom=NULL;
	}
	
		
	//Normalize input matrices
	cblas_dcopy(f*o,Xini,1,X,1);
	dnrm1ColsSave(f, o, X, normsX);
	cblas_dcopy(f*n,Bini,1,B,1);
	dnrm2ColsSave(f, n, B, normsB);
	
	daddEsc_x86(f*o, EPS, X);

	//precomputing Bt and fixed part of eq. 11
	dtrans(f, n, B, Bt);
	dmultOnes(f,n,o,Bt,BtO);
	
	//Computing initial atoms and adding them to the active set
	compInitialAtoms(f,n,o,X,B,initAtoms, initWeights);
	for(j = 0; j < o; j++)
	{
		addAtom(&activeObs[j],initAtoms[j]);
        
		W[initAtoms[j]+ j*n] = initWeights[j];
       // if (j < 10)
         //   printf("%d(%g) ",initAtoms[j],initWeights[j]);
	}
	
	//Perform algoritm iterations (starts with 2 for equivalence with the original version)	
	for(it=2; it < iter; it++)
	{
		//V = activeB * activeW
		for(j = 0; j < nObs; j++)
			if( !activeObs[j].converged)
			{
				for(i =0; i < f; i ++)
				{
					V[i + j*f] =0;
					for(k = 0, atomPtr = activeObs[j].firstAtom; k <  activeObs[j].nAtoms;k++)
					{
						V[i + j*f] += B[i + atomPtr->atomIdx * f] * W[ atomPtr->atomIdx + j*n];
						
						if( k != activeObs[j].nAtoms -1)
							atomPtr = atomPtr->nextAtom;
					}
				}
			}

		//R = X/V
		ddiv_x86(f*o,X,V,R);

		//Each 2 iterations try to add new atoms
		if(it % 2 == 0)
		{
			
			cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,o,f,1,Bt,n,R,f,0,gradAll,n);
			dsub_x86(n*o,BtO,gradAll,gradAll);//TODO try to change dgemm+dsub with dcopy+dgemm
			gradientComputed=1;
			cblas_dcopy(n*o,gradAll,1,gradAll2,1);//TODO try to remove this dcopy
			
			
			//Check convergence
			if(it % 10 == 0)
			{
				converged = 0;
				for(i = 0; i < n*o; i++)
					if(gradAll2[i] > 0)
						gradAll2[i] = 0;
				
				//Mark each converged observation
				for(j=0; j < o; j++)
					if( !activeObs[j].converged)
					{
						norm = cblas_ddot(n,&gradAll2[j*n],1,&gradAll2[j*n],1);
						if(norm < 1e-15)
						{
							activeObs[j].converged = 1;
							converged++;
						}
					}
					else
						converged++;
				
				//If all observations are converged finish the algorithm
				if(converged == o)
				{
					//show error previous to scaling
					//printf("inner Error on it %d: %g\n",it,derrorAsna(f,o,n,X,B,W));
					//rescale back to original scale of S
					freeObservations(nObs,activeObs);
                    drowsRescale(n, o, W, normsB);
					dcolsRescale(n, o, W, normsX);
                        #ifdef With_MKL
                            mkl_free(X);
                            mkl_free(B);
                            mkl_free(Bt);
                            mkl_free(BtO);
                            mkl_free(V);
                            mkl_free(R);
                            mkl_free(gradAll);
                            mkl_free(gradAll2);
                            mkl_free(normsX);
                            mkl_free(normsB);
                            mkl_free(initAtoms);
                            mkl_free(initWeights);
                            mkl_free(activeObs);
                            mkl_free(Mult);
                            mkl_free(Hwa);
                            mkl_free(Aux);
                            mkl_free(grad);
                            mkl_free(maxStep);
                        #else
                            _mm_free(X);
                            _mm_free(B);
                            _mm_free(Bt);
                            _mm_free(BtO);
                            _mm_free(V);
                            _mm_free(R);
                            _mm_free(gradAll);
                            _mm_free(gradAll2);
                            _mm_free(normsX);
                            _mm_free(normsB);
                            _mm_free(initAtoms);
                            _mm_free(initWeights);
                            _mm_free(activeObs);
                            _mm_free(Mult);
                            _mm_free(Hwa);
                            _mm_free(Aux);
                            _mm_free(grad);
                            _mm_free(maxStep);
                        #endif

					return 0;
				}
			}
			
			//add only entries that are not yet active
			for(j = 0; j < nObs; j++)
				if( !activeObs[j].converged)
				{
					for(i = 0, atomPtr = activeObs[j].firstAtom; i <  activeObs[j].nAtoms;i++)
					{
						gradAll2[atomPtr->atomIdx + j*n] = 0;
						
						if( i != activeObs[j].nAtoms -1)
							atomPtr = atomPtr->nextAtom;
					}			

				}
				
			//Adding atoms
			for(j = 0; j < nObs; j++)
				if( !activeObs[j].converged)
				{
					minIdx = dMin_x86(n,&gradAll2[j*n]);
					if(gradAll2[minIdx + j *n] < -1e-10)
					{
						addAtom(&activeObs[j],minIdx);		
						W[minIdx + j*n] = gradAll2[minIdx + j *n] < 1e-15? 1e-15: gradAll2[minIdx + j *n]; 
					}

				}
		}
		else
			gradientComputed = 0;
		

		//Computing each non-converged observation
		for(j = 0; j < nObs; j++)
			if( !activeObs[j].converged)
			{
				//Compute or copy gradients and the hessian matrix
				for(i = 0, atomPtr = activeObs[j].firstAtom; i <  activeObs[j].nAtoms;i++)
				{
					dmult_x86(f,&B[atomPtr->atomIdx*f],&R[j*f],Mult);
					if(gradientComputed)//{
						grad[i] = gradAll[atomPtr->atomIdx + j*n];
					else
						grad[i] = BtO[atomPtr->atomIdx + j*n] - cblas_dasum(f,Mult,1);

					ddiv_x86(f,Mult,&V[j*f],Aux);
					for(k=i, atomPtr2= atomPtr; k < activeObs[j].nAtoms; k++ )
					{

						Hwa[ k + i * nnz] = cblas_ddot(f,&B[atomPtr2->atomIdx *f],1,Aux,1);
						if(k == i)
							Hwa[k + i*nnz] +=1e-10; 
						
						if( k != activeObs[j].nAtoms -1)
							atomPtr2 = atomPtr2->nextAtom;
					}
					
					if( i != activeObs[j].nAtoms -1)
						atomPtr = atomPtr->nextAtom;
				}
				
				
				//Solve Hwa * x = grad;
 /*              if(it == 2 && j==0)
                {
                    printf("atoms %d, n %d\n",activeObs[j].nAtoms,n);
                    printf("atoms %d %d\n",activeObs[j].firstAtom->atomIdx, activeObs[j].lastAtom -> atomIdx);
                    for(k = 0; k < activeObs[j].nAtoms; k++)
                    {
                        for(k2 = 0; k2 < activeObs[j].nAtoms; k2++)
                        {
                            printf("%g ",Hwa[k+k2*nnz]);
                        }
                        printf("\n");
                    }
                }*/
				info = LAPACKE_dpotrf(LAPACK_COL_MAJOR,'L',activeObs[j].nAtoms,Hwa,nnz);
				if (info != 0){
					printf("Error en lapack(dpotrf): %d it=%d, obs=%d\n",info,it,j);
                }
				info = LAPACKE_dpotrs(LAPACK_COL_MAJOR,'L',activeObs[j].nAtoms,1,Hwa,nnz,grad,n);
				if (info != 0)
					printf("Error en lapack(dpotrs): %d\n",info);
					
				//Computing step direction
				for(i = 0, atomPtr = activeObs[j].firstAtom; i <  activeObs[j].nAtoms;i++)
				{
					if(grad[i]<=0)
						maxStep[i] = HUGE_VAL;
					else
						maxStep[i] = W[atomPtr->atomIdx + n*j] / grad[i];
					
					if( i != activeObs[j].nAtoms -1)
						atomPtr = atomPtr->nextAtom;
				}
				
				//Computing step size
				//minIdx = cblas_idamin(activeObs[j].nAtoms,maxStep,1);//TODO check idamin vs dmin_x86
				minIdx = dMin_x86(activeObs[j].nAtoms,maxStep);
				stepSize = maxStep[minIdx];
				if( stepSize < 0 || stepSize > 1)
					stepSize = 1;

				
				//Update weigths and remove negative or zero weigths from the active set 
				for(i = 0,gPtr=0, atomPtr = activeObs[j].firstAtom; i < activeObs[j].nAtoms; i++,gPtr++)
				{
					num = W[atomPtr->atomIdx + n*j] - stepSize * grad[gPtr];
				
					if(num <= EPS) 
					{
						W[atomPtr->atomIdx + n*j] = 0;
						if( i != activeObs[j].nAtoms -1 )
						{
							atomPtr2 = atomPtr->nextAtom;
							removeAtom(&activeObs[j],atomPtr->atomIdx);
							atomPtr = atomPtr2;
							i--;
						}
						else
							removeAtom(&activeObs[j],atomPtr->atomIdx);
					}
					else
					{
						W[atomPtr->atomIdx + n*j] = num;
						
						if( i != activeObs[j].nAtoms -1 )
							atomPtr = atomPtr->nextAtom;
					}

				}
			}
		
	}
	
	freeObservations(nObs,activeObs);
    drowsRescale(n, o, W, normsB);
    dcolsRescale(n, o, W, normsX);
    
    #ifdef With_MKL
		mkl_free(X);
		mkl_free(B);
        mkl_free(Bt);
		mkl_free(BtO);
		mkl_free(V);
		mkl_free(R);
		mkl_free(gradAll);
		mkl_free(gradAll2);
		mkl_free(normsX);
		mkl_free(normsB);
		mkl_free(initAtoms);
		mkl_free(initWeights);
		mkl_free(activeObs);
		mkl_free(Mult);
		mkl_free(Hwa);
		mkl_free(Aux);
		mkl_free(grad);
		mkl_free(maxStep);
	#else
		_mm_free(X);
		_mm_free(B);
        _mm_free(Bt);
		_mm_free(BtO);
		_mm_free(V);
		_mm_free(R);
		_mm_free(gradAll);
		_mm_free(gradAll2);
		_mm_free(normsX);
		_mm_free(normsB);
		_mm_free(initAtoms);
		_mm_free(initWeights);
		_mm_free(activeObs);
		_mm_free(Mult);
		_mm_free(Hwa);
		_mm_free(Aux);
		_mm_free(grad);
		_mm_free(maxStep);
	#endif

        
    return 0;
}

/**
 *  \fn   int dasna_cpu(const int f, const int n, const int o, const double* Xini, const double* Bini, double* W ,const int iter)
 *  \brief This function performs a matrix nonnegative approximation using Active-Set Newton Algorithm (ASNA) X = BW using parallel computation techniques
 *  \param f: (input) Number of features
 *  \param n: (input) Number of dictionary atoms
 *  \param o: (input) Number of observations
 *  \param Xini: (input) Double precision matrix to factorize (f x o)(1D column-major)
 *  \param Bini: (input) Double precision dictionary matrix (f x n) (1D column-major)
 *  \param W: (output) Double precision weigths matrix (n x o) (1D column-major)
 *  \param iter: (input)Maximum number of iterations to perform
 *  \param nnz: (input) Maximum number of active atoms per observation (to restrict memory size of the hessian matrix, maximum n)
*/
int dasnaPar_cpu(const int f, const int n, const int o, const double* Xini, const double* Bini, double* W ,const int iter, const int nnz)//MOD
{
  
	int i,j,k,it, info, gPtr;
	int nThreads, tid;
	int gradientComputed, minIdx, converged, nObs;
	int *initAtoms;
	double stepSize, num,norm;
    double *B, *X, *Bt, *BtO,*V, *R, *gradAll, *gradAll2, *Hwa;
    double *normsX, *normsB, *initWeights, *Mult, *Aux, *grad, *maxStep;
	double *MultS, *AuxS, *gradS, *maxStepS, *HwaS;
    observation* activeObs, *obsPtr;
	atom* atomPtr, *atomPtr2;
	
	#ifdef With_MKL
        X = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
		B = (double *)mkl_malloc(f * n * sizeof(double), WRDLEN);
		Bt = (double *)mkl_malloc(n * f * sizeof(double), WRDLEN);
		BtO = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
		V = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
		R = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
		gradAll = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
		gradAll2 = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
		normsX = (double *)mkl_malloc(o * sizeof(double), WRDLEN);
		normsB = (double *)mkl_malloc(n * sizeof(double), WRDLEN);
		initAtoms = (int *)mkl_malloc(o * sizeof(int), WRDLEN);
		initWeights = (double *)mkl_malloc(o * sizeof(double), WRDLEN);
		activeObs = (observation*)mkl_malloc(o*sizeof(observation),WRDLEN);
    #else
        X = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
		B = (double *)_mm_malloc(f * n * sizeof(double), WRDLEN);
		Bt = (double *)_mm_malloc(n * f * sizeof(double), WRDLEN);
		BtO = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
		V = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
		R = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
		gradAll = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
		gradAll2 = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
		normsX = (double *)_mm_malloc(o * sizeof(double), WRDLEN);
		normsB = (double *)_mm_malloc(n * sizeof(double), WRDLEN);
		initAtoms = (int *)_mm_malloc(o * sizeof(int), WRDLEN);
		initWeights = (double *)_mm_malloc(o * sizeof(double), WRDLEN);
		activeObs = (observation *)_mm_malloc(o * sizeof(observation), WRDLEN);
    #endif
	
	//allocation for false sharing private thread variables
	nThreads = omp_get_max_threads();
	#ifdef With_MKL
		MultS = (double *)mkl_malloc(nThreads * f * sizeof(double), WRDLEN);
		AuxS = (double *)mkl_malloc(nThreads * f * sizeof(double), WRDLEN);
		gradS = (double *)mkl_malloc(nThreads * n * sizeof(double), WRDLEN);
		maxStepS = (double *)mkl_malloc(nThreads * n * sizeof(double), WRDLEN);
		HwaS = (double *)mkl_malloc(nThreads * nnz *nnz * sizeof(double), WRDLEN);
	#else
		MultS = (double *)_mm_malloc(nThreads * f * sizeof(double), WRDLEN);
		AuxS = (double *)_mm_malloc(nThreads * f * sizeof(double), WRDLEN);
		gradS = (double *)_mm_malloc(nThreads * n * sizeof(double), WRDLEN);
		maxStepS = (double *)_mm_malloc(nThreads * n * sizeof(double), WRDLEN);
		HwaS = (double *)_mm_malloc(nThreads * nnz *nnz * sizeof(double), WRDLEN);
	#endif
    
    if(HwaS == NULL)
        printf("Error allocating the hessian matrix\n");

	//init observation structure
	nObs = o;
	for(i = 0; i < o  ; i++)
	{
		activeObs[i].obsIdx=i;
		activeObs[i].converged=0;
	    activeObs[i].nAtoms=0;
		activeObs[i].firstAtom=NULL;
		activeObs[i].lastAtom=NULL;
	}
	
		
	//Normalize input matrices
	cblas_dcopy(f*o,Xini,1,X,1);
	dnrm1ColsSave(f, o, X, normsX);
	cblas_dcopy(f*n,Bini,1,B,1);
	dnrm2ColsSave(f, n, B, normsB);
	
	daddEsc_x86(f*o, EPS, X);

	//precomputing Bt and fixed part of eq. 11
	dtrans(f, n, B, Bt);
	dmultOnes(f,n,o,Bt,BtO);
	
	//Computing initial atoms and adding them to the active set
	compInitialAtoms(f,n,o,X,B,initAtoms, initWeights);
	#pragma omp parallel for schedule(static)
	for(j = 0; j < o; j++)
	{
		addAtom(&activeObs[j],initAtoms[j]);
		W[initAtoms[j]+ j*n] = initWeights[j];
	}
	
	//Perform algoritm iterations (starts with 2 for equivalence with the original version)		
	for(it=2; it < iter; it++)
	{
		//V = activeB * activeW
		#pragma omp parallel for  schedule(dynamic) private(i,k,atomPtr)
		for(j = 0; j < nObs; j++)
			if( !activeObs[j].converged)
			{
				for(i =0; i < f; i ++)
				{
					V[i + j*f] =0;
					for(k = 0, atomPtr = activeObs[j].firstAtom; k <  activeObs[j].nAtoms;k++)
					{
						V[i + j*f] += B[i + atomPtr->atomIdx * f] * W[ atomPtr->atomIdx + j*n];
					
						if( k != activeObs[j].nAtoms -1)
							atomPtr = atomPtr->nextAtom;
					}
				}
			}

		//R = X/V
		ddiv_x86(f*o,X,V,R);
		
		//Each 2 iterations try to add new atoms
		if(it % 2 == 0)
		{
			cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,o,f,1,Bt,n,R,f,0,gradAll,n);
			dsub_x86(n*o,BtO,gradAll,gradAll);//TODO try to change dgemm+dsub with dcopy+dgemm
			gradientComputed=1;
			cblas_dcopy(n*o,gradAll,1,gradAll2,1);//TODO try to remove this dcopy
			
			
			//Check convergence
			if(it % 10 == 0)
			{
				converged = 0;
				#pragma omp parallel for schedule(static)
				for(i = 0; i < n*o; i++)
					if(gradAll2[i] > 0)
						gradAll2[i] = 0;
				
				//Mark each converged observation
				for(j=0; j < o; j++)
					if( !activeObs[j].converged)
					{
						norm = cblas_ddot(n,&gradAll2[j*n],1,&gradAll2[j*n],1);
						if(norm < 1e-15)
						{
							activeObs[j].converged = 1;
							converged++;
						}
					}
					else
						converged++;
				
				//If all observations are converged finish the algorithm
				if(converged == o)
				{
					//show error previous to scaling
					//printf("inner Error on it %d: %g\n",it,derrorAsna(f,o,n,X,B,W));
					//rescale back to original scale of S
					freeObservations(nObs,activeObs);
                    drowsRescale(n, o, W, normsB);
					dcolsRescale(n, o, W, normsX);
                        #ifdef With_MKL
                            mkl_free(X);
                            mkl_free(B);
                            mkl_free(Bt);
                            mkl_free(BtO);
                            mkl_free(V);
                            mkl_free(R);
                            mkl_free(gradAll);
                            mkl_free(gradAll2);
                            mkl_free(normsX);
                            mkl_free(normsB);
                            mkl_free(initAtoms);
                            mkl_free(initWeights);
                            mkl_free(activeObs);
                            mkl_free(MultS);
                            mkl_free(gradS);
                            mkl_free(AuxS);
                            mkl_free(maxStepS);
                            mkl_free(HwaS);
                        #else
                            _mm_free(X);
                            _mm_free(B);
                            _mm_free(Bt);
                            _mm_free(BtO);
                            _mm_free(V);
                            _mm_free(R);
                            _mm_free(gradAll);
                            _mm_free(gradAll2);
                            _mm_free(normsX);
                            _mm_free(normsB);
                            _mm_free(initAtoms);
                            _mm_free(initWeights);
                            _mm_free(activeObs);
                            _mm_free(MultS);
                            _mm_free(gradS);
                            _mm_free(AuxS);
                            _mm_free(maxStepS);
                            _mm_free(HwaS);
                        #endif

					return 0;
				}
			}
			
			
			//add only entries that are not yet active
			#pragma omp parallel for schedule(dynamic) private(i,atomPtr)
			for(j = 0; j < nObs; j++)
				if( !activeObs[j].converged)
				{
					for(i = 0, atomPtr = activeObs[j].firstAtom; i <  activeObs[j].nAtoms;i++)
					{
						gradAll2[atomPtr->atomIdx + j*n] = 0;
						
						if( i != activeObs[j].nAtoms -1)
							atomPtr = atomPtr->nextAtom;
					}			

				}
			//Adding atoms
			#pragma omp parallel for schedule(dynamic) private(minIdx)
			for(j = 0; j < nObs; j++)
				if( !activeObs[j].converged)
				{
					minIdx = dMin_x86(n,&gradAll2[j*n]);
					if(gradAll2[minIdx + j *n] < -1e-10)
					{
						addAtom(&activeObs[j],minIdx);
						W[minIdx + j*n] = gradAll2[minIdx + j *n] < 1e-15? 1e-15: gradAll2[minIdx + j *n]; 
					}

				}
		}
		else
			gradientComputed = 0;
		
		//Computing each non-converged observation in parallel
		#pragma omp parallel default(none)  shared(nObs,activeObs,B,R,gradientComputed,gradAll,BtO,V,it,W, MultS,AuxS, maxStepS, gradS,HwaS,nThreads) private(i,atomPtr,Mult, grad,Aux,k,atomPtr2,Hwa,info,maxStep,minIdx,stepSize,gPtr,num,tid)
		{
			
			tid = omp_get_thread_num();
			Mult = &MultS[ tid* f];
			Aux = &AuxS[ tid* f];
			maxStep = &maxStepS[ tid* n];
			grad = &gradS[ tid* n];
			Hwa = &HwaS[(unsigned long)tid * nnz*nnz];
			
			#pragma omp for schedule(dynamic) 
			for(j = 0; j < nObs; j++)
				if( !activeObs[j].converged)
				{
					//Compute or copy gradients and the hessian matrix
					for(i = 0, atomPtr = activeObs[j].firstAtom; i <  activeObs[j].nAtoms;i++)
					{
						dmult_x86(f,&B[atomPtr->atomIdx*f],&R[j*f],Mult);
						if(gradientComputed)
							grad[i] = gradAll[atomPtr->atomIdx + j*n];
						else
							grad[i] = BtO[atomPtr->atomIdx + j*n] - cblas_dasum(f,Mult,1);
						
						ddiv_x86(f,Mult,&V[j*f],Aux);
						for(k=i, atomPtr2= atomPtr; k < activeObs[j].nAtoms; k++ )
						{
							Hwa[ k + i * nnz] = cblas_ddot(f,&B[atomPtr2->atomIdx *f],1,Aux,1);
							if(k == i)
								Hwa[k + i*nnz] +=1e-10; 
							
							if( k != activeObs[j].nAtoms -1)
								atomPtr2 = atomPtr2->nextAtom;
						}
						
						if( i != activeObs[j].nAtoms -1)
							atomPtr = atomPtr->nextAtom;
					}
					
					
					//Solve Hwa * x = grad;
					info = LAPACKE_dpotrf(LAPACK_COL_MAJOR,'L',activeObs[j].nAtoms,Hwa,nnz);
					if (info != 0)
						printf("Error en lapack(dpotrf): %d it=%d, obs=%d\n",info,it,j);
					info = LAPACKE_dpotrs(LAPACK_COL_MAJOR,'L',activeObs[j].nAtoms,1,Hwa,nnz,grad,n);
					if (info != 0)
						printf("Error en lapack(dpotrs): %d\n",info);
						
					//Computing step direction
					for(i = 0, atomPtr = activeObs[j].firstAtom; i <  activeObs[j].nAtoms;i++)
					{
						if(grad[i]<=0)
							maxStep[i] = HUGE_VAL;
						else
							maxStep[i] = W[atomPtr->atomIdx + n*j] / grad[i];
						
						if( i != activeObs[j].nAtoms -1)
							atomPtr = atomPtr->nextAtom;
					}
					
					//Computing step size
					//minIdx = cblas_idamin(activeObs[j].nAtoms,maxStep,1);
					minIdx = dMin_x86(activeObs[j].nAtoms,maxStep);
					stepSize = maxStep[minIdx];
					if( stepSize < 0 || stepSize > 1)
						stepSize = 1;

					
					//Update weigths and remove negative or zero weigths from the active set 
					for(i = 0,gPtr=0, atomPtr = activeObs[j].firstAtom; i < activeObs[j].nAtoms; i++,gPtr++)
					{
						num = W[atomPtr->atomIdx + n*j] - stepSize * grad[gPtr];

						if(num <= EPS) 
						{
							W[atomPtr->atomIdx + n*j] = 0;
							if( i != activeObs[j].nAtoms -1 )
							{
								atomPtr2 = atomPtr->nextAtom;
								removeAtom(&activeObs[j],atomPtr->atomIdx);
								atomPtr = atomPtr2;
								i--;
							}
							else
								removeAtom(&activeObs[j],atomPtr->atomIdx);
						}
						else
						{
							W[atomPtr->atomIdx + n*j] = num;
							
							if( i != activeObs[j].nAtoms -1 )
								atomPtr = atomPtr->nextAtom;
						}
						
					}
				}
		}

	}
	
	freeObservations(nObs,activeObs);
    
    drowsRescale(n, o, W, normsB);
    dcolsRescale(n, o, W, normsX);
    #ifdef With_MKL
		mkl_free(X);
		mkl_free(B);
		mkl_free(Bt);
		mkl_free(BtO);
		mkl_free(V);
		mkl_free(R);
		mkl_free(gradAll);
		mkl_free(gradAll2);
		mkl_free(normsX);
		mkl_free(normsB);
		mkl_free(initAtoms);
		mkl_free(initWeights);
		mkl_free(activeObs);
		mkl_free(MultS);
		mkl_free(gradS);
		mkl_free(AuxS);
		mkl_free(maxStepS);
		mkl_free(HwaS);
	#else
		_mm_free(X);
		_mm_free(B);
		_mm_free(Bt);
		_mm_free(BtO);
		_mm_free(V);
		_mm_free(R);
		_mm_free(gradAll);
		_mm_free(gradAll2);
		_mm_free(normsX);
		_mm_free(normsB);
		_mm_free(initAtoms);
		_mm_free(initWeights);
		_mm_free(activeObs);
		_mm_free(MultS);
		_mm_free(gradS);
		_mm_free(AuxS);
		_mm_free(maxStepS);
		_mm_free(HwaS);
	#endif


        
    return 0;
}


/**
 *  \fn   void dnrm1ColsSave(const int m, const int n, double* __restrict__ M, double* __restrict__ norms)
 *  \brief This function performs a column 1-norm normalization and returns the column norms
 *  \param m: (input) Number of rows of M
 *  \param n: (input) Number of cols of M
 *  \param M: (inout) Double precision input/output matrix (1D column-major)
 *  \param norms: (output) Double precision output vector of size n
*/
void dnrm1ColsSave(const int m, const int n, double* __restrict__ M, double* __restrict__ norms)
{
    int i;
    
    #ifdef With_ICC
      #pragma loop_count min=32
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for( i = 0; i < n; i++)
    {
        norms[i] = cblas_dasum(m,&M[i*m],1) + EPS;
        cblas_dscal(m,1.0/norms[i],&M[i*m],1);
    }
}

/**
 *  \fn   void dnrm2ColsSave(const int m, const int n, double* __restrict__ M,  double* __restrict__ norms)
 *  \brief This function performs a column 2-norm normalization and returns the column norms
 *  \param m: (input) Number of rows of M
 *  \param n: (input) Number of cols of M
 *  \param M: (inout) Double precision input/output matrix (1D column-major)
 *  \param norms: (output) Double precision vector of size n
*/
void dnrm2ColsSave(const int m, const int n, double* __restrict__ M,  double* __restrict__ norms)
{
    int i;
    
    #ifdef With_ICC
      #pragma loop_count min=32
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for( i = 0; i < n; i++)
    {
        norms[i] = cblas_dnrm2(m,&M[i*m],1) + EPS;
        cblas_dscal(m,1.0/norms[i],&M[i*m],1);
    }

}

/**
 *  \fn   void dcolsRescale(const int m, const int n, double* __restrict__ M, const double* __restrict__ norms)
 *  \brief This function performs a column scale back of matrix M using the original norms contained in norms 
 *  \param m: (input) Number of rows of M
 *  \param n: (input) Number of cols of M
 *  \param M: (inout) Double precision input/output matrix (1D column-major)
 *  \param norms: (input) Double precision vector of size n
*/
void dcolsRescale(const int m, const int n, double* __restrict__ M, const double* __restrict__ norms)
{
    int i;
    
    #ifdef With_ICC
      #pragma loop_count min=32
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for( i = 0; i < n; i++)
        cblas_dscal(m,norms[i],&M[i*m],1);
}

/**
 *  \fn   void drowsRescale(const int m, const int n, double* __restrict__ M, const double* __restrict__ norms)
 *  \brief This function performs a row scale back of matrix M using the original norms contained in norms 
 *  \param m: (input) Number of rows of M
 *  \param n: (input) Number of cols of M
 *  \param M: (inout) Double precision input/output matrix (1D column-major)
 *  \param norms: (input) Double precision vector of size m
*/
void drowsRescale(const int m, const int n, double* __restrict__ M, const double* __restrict__ norms)
{
    int i;
    
    #ifdef With_ICC
      #pragma loop_count min=32
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for( i = 0; i < m; i++)
        cblas_dscal(n,norms[i],&M[i],m);
}

/**
 *  \fn    void dtrans(const int m, const int n, const double* M, double* __restrict__ Mt)
 *  \brief This function performs copy and transposition of a double precision matrix Mt = M'
 *  \param m: (input) Number of rows of M
 *  \param n: (input) Number of cols of M
 *  \param M: (input) Double precision matrix (1D column-major)
 *  \param Mt: (output) Double precision matrix (1D column-major)
*/
void dtrans(const int m, const int n, const double* M, double* __restrict__ Mt)
{
  //#ifdef With_MKL Comentado temporalmente porque no funciona en mkl de heybulldog
  //  mkl_domatcopy('C','T',m,n,1,M,m,Mt,n);
  //#else
    int i, j;
    #ifdef With_ICC
      #pragma loop_count min=512
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for (j = 0; j < n; j++)
        for (i = 0; i < m; i++)
            Mt[j+i*n] = M[i+j*m];
  //#endif
}

/**
 *  \fn    void daddEsc_x86(const int n, const double x, double *__restrict__ y)
 *  \brief This function performs double precision scalar-vector adition y[i]=x+y[i]
 *  \param n: (input) Number of elements of  y
 *  \param x: (input) Double precision scalar
 *  \param y: (inout) Double precision input/output vector/matrix (1D column-major)
*/
void daddEsc_x86(const int n, const double x, double *__restrict__ y)
{
    int i;
    #ifdef With_ICC
      #pragma loop_count min=512
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for (i=0; i<n; i++)
      y[i] = x + y[i];

}

/**
 *  \fn   void ddiv_x86(const int n, const double *x, const double y,  double *__restrict__ z)
 *  \brief This function calls the appropiate funtions to performs double precision
 *         element-wise  z[i]=x[i]/y[i] for all positions of x and y
 *  \param n: (input) Number of elements of x, y and z
 *  \param x: (input) Double precision input vector/matrix (1D column-major)
 *  \param y: (input) Double precision input vector/matrix (1D column-major)
 *  \param z: (inout) Double precision input/output vector/ matrix (1D column-major)
*/
void ddiv_x86(const int n, const double *x, const double* y,  double *__restrict__ z)
{
  #ifdef With_MKL
    vdDiv(n, x, y, z);
  #else
    int i;
    #ifdef With_ICC
      #pragma loop_count min=32
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for (i=0; i<n; i++)
    {
      #ifdef With_Check
        /* Here we can have NaN and Inf if y(i) and/or x(i)=0 */
        z[i]=x[i] / y[i];
        assert(isfinite(y[i]));
      #else
        z[i]=x[i] / y[i];
      #endif
    }
  #endif
}

/**
 *  \fn   void dmult_x86(const int n, const double *x, const double y,  double *__restrict__ z)
 *  \brief This function calls the appropiate funtions to performs double precision
 *         element-wise  z[i]=x[i]*[i] for all positions of x and y
 *  \param n: (input) Number of elements of x and y
 *  \param x: (input) Double precision input vector/matrix (1D column-major)
 *  \param y: (input) Double precision input vector/matrix (1D column-major)
 *  \param z: (inout) Double precision input/output vector/ matrix (1D column-major)
*/
void dmult_x86(const int n, const double *x, const double *y,  double *__restrict__ z)
{
  #ifdef With_MKL
    vdMul(n, x, y, z);
  #else
    int i;
    #ifdef With_ICC
      #pragma loop_count min=32
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for (i=0; i<n; i++)
    {
        z[i]=x[i] * y[i];
    }
  #endif
}

/**
 *  \fn    void dsub_x86(const int n, const double *x, const double *y, double *__restrict__ z)
 *  \brief This function performs double precision element-wise substraction z[i]=x[i]-y[i]
 *  \param n: (input) Number of elements of x, y and z
 *  \param x: (input) Double precision input vector/matrix (1D column-major)
 *  \param y: (input) Double precision input vector/matrix (1D column-major)
 *  \param z: (inout) Double precision input/output vector/ matrix (1D column-major)
*/
void dsub_x86(const int n, const double *x, const double *y, double *__restrict__ z)
{
  #ifdef With_MKL
    vdSub(n, x, y, z);
  #else
    int i;
    #ifdef With_ICC
      #pragma loop_count min=512
      #pragma simd
    #else
      #pragma omp parallel for
    #endif
    for (i=0; i<n; i++)
      /* ask for x[i] or y[i] = 0.0 don't give improvements. We don't do it */
      z[i] = x[i] - y[i];
  #endif
}

/**
 *  \fn   void dmemset_x86(const int n, double *__restrict__ x, const double val)
 *  \brief This function fills all positions of x with val
 *  \param n:   (input)  Number of elements of x
 *  \param x:   (output) Double precision output matrix (1D column-major) or vector
 *  \param val: (input)  Double precision value
*/
void dmemset_x86(const int n, double *__restrict__ x, const double val)
{
  int i;
  
  #ifdef With_ICC
    #pragma loop_count min=1024
    #pragma simd
  #else
    #pragma omp parallel for
  #endif
  for (i=0; i<n; i++)
    x[i]=val;
}

/**
 *  \fn   void imemset_x86(const int n, int *__restrict__ x, const int val)
 *  \brief This function fills all positions of x with val
 *  \param n:   (input)  Number of elements of x
 *  \param x:   (output) Integer output matrix (1D column-major) or vector
 *  \param val: (input)  Integer value
*/
void imemset_x86(const int n, int *__restrict__ x, const int val)
{
  int i;

  #ifdef With_ICC
    #pragma loop_count min=1024
    #pragma simd
  #else
    #pragma omp parallel for
  #endif
  for (i=0; i<n; i++)
    x[i]=val;
}

/**
 *  \fn    int dMin_x86(const int n, const double* x)
 *  \brief This function computes the minimum element of a vector
 *  \param n: (input) Number of elements of x
 *  \param x: (input) Double precision input vector/matrix (1D column-major)
 *  \return integer index of minimum element of X
*/
int dMin_x86(const int n, const double* __restrict__ x)
{
	int i,minIdx = 0;
	double minVal = x[0];
	
	#pragma omp parallel
	{
		int minIdxLocal = minIdx;
		double minLocal = minVal;
		#pragma omp for nowait
		for(i = 1; i < n; i++)
			if(x[i] < minLocal)
			{
					minLocal = x[i];
					minIdxLocal = i;
			}
			
		#pragma omp critical
		{
			if(minLocal < minVal)
			{
				minVal = minLocal;
				minIdx = minIdxLocal;
			}
			
		}

	}	
	return minIdx;	
	
}

/**
 *  \fn    void derrorbd1_x86(const int n, const double *x, double *__restrict__ y)
 *  \brief This function performs auxiliar double precision operations when error is computed
           using betadivergence error formula and beta = 1
 *  \param n: (input) Number of elements of x and y
 *  \param x: (input) Double precision input vector/matrix
 *  \param y: (inout) Double precision input/output vector/matrix
*/
void derrorbd1_x86(const int n, const double *x, double *__restrict__ y)
{
  int i;

  #ifdef With_ICC
    #pragma loop_count min=16
    #pragma simd
  #else
    #pragma omp parallel for
  #endif
  for (i=0; i<n; i++)
  {
    #ifdef With_Check
      /* Here we can have NaN and Inf if y(i) and/or x(i)=0 */
      y[i]=(x[i]*log(x[i]/y[i])) + y[i] - x[i];
      assert(isfinite(y[i]));
    #else
      y[i]=(x[i]*log(x[i]/y[i])) + y[i] - x[i];
    #endif
  }
}
