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
 ***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <assert.h>

#ifdef With_MKL
  #include <mkl.h>
#else
  #include <cblas.h>
  #include <lapacke.h>
  #include <mm_malloc.h>
  extern void dlarnv_(int *, int *, int *, double *);
  extern void slarnv_(int *, int *, int *, float  *);
#endif

#define WRDLEN 32

#define max(a,b) (((a)>(b ))?( a):(b))
#define min(a,b) (((a)<(b ))?( a):(b))

#define EPS 1e-16 //litle value to avoid zeros 


typedef struct atom atom;
typedef struct observation observation;

//Structure to keep each obvservation information
struct observation
{
	int obsIdx; 
	int converged; // 0 if the observation is not converged
	int nAtoms; //number of active atoms in the observation
	atom* firstAtom; //First element in the double linked list of active atoms
	atom* lastAtom; //First element in the double linked list of active atoms
};

//Element of double linked list of atoms
struct atom
{
	int atomIdx; //Position of atom in the observation
	atom* prevAtom;
	atom* nextAtom;
};


void dmultOnes(const int f, const int n, const int o, const double* M, double* __restrict__ MO);
void compInitialAtoms(const int f, const int n, const int o, const double* X, const double* B, int* __restrict__ initAtoms, double* __restrict__ initWeigths);
void addAtom(observation* obs, const int newIdx);
void removeAtom(observation* obs, const int remIdx);
double derrorAsna(const int m, const int n, const int k, const double *A, const double *W, const double *H);//MOD


/*ASNA implementations */
int dasna_cpu(const int f, const int n, const int o,  const double* X,  const double* B, double* W ,const int iter, const int nnz);
int dasnaPar_cpu(const int f, const int n, const int o, const double* Xini, const double* Bini, double* W ,const int iter, const int nnz);

/*Common auxiliar functions */

void dnrm1ColsSave(const int m, const int n, double* __restrict__ M, double* __restrict__ norms); 
void dnrm2ColsSave(const int m, const int n, double* __restrict__ M, double* __restrict__ norms); 
void dcolsRescale(const int m, const int n, double* __restrict__ M, const double* __restrict__ norms); 
void drowsRescale(const int m, const int n, double* __restrict__ M, const double* __restrict__ norms); 
void dtrans(const int m, const int n, const double* M, double* __restrict__ Mt); 
void ddiv_x86(const int n, const double *x, const double* y,  double *__restrict__ z);
void dmult_x86(const int n, const double *x, const double* y,  double *__restrict__ z);
void dsub_x86(const int n, const double *x, const double *y, double *__restrict__ z);
void daddEsc_x86(const int n, const double x, double *__restrict__ y);
void dmemset_x86(const int n, double *__restrict__ x, const double val);
void imemset_x86(const int n, int  *__restrict__ x, const int  val);
int dMin_x86(const int n, const double* __restrict__ x);
void derrorbd1_x86(const int n, const double *x, double *__restrict__ y);
