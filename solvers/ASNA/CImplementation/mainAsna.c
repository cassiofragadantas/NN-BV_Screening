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

#include "asna.h"

int dread_file(char *path, int rows, int cols, double *M) ;
int dwrite_file(char *path, int rows, int cols, double *M); 
int read_file_header(char *path, int *rows, int *cols);

int main(int argc, char *argv[])
{
  int
    f,      /* number of rows   of matrix X */
    o,      /* number of colums of matrix X */
    n,      /* number of colums of matrix B and number of rows of matrix W */
    status,
    iters,
    seed,
	repe,
	i,
    nnz =500;

  double
    *X=NULL,
    *Xr=NULL,
    *B=NULL,
    *W=NULL,
    *B0=NULL,
    *W0=NULL,
    *w0=NULL,
    error=0.0,
    Time =0.0,
	t = 0.0;

  char
      *inX = NULL,
      *inB = NULL;
  if (argc != 5)
  {
    printf("Apply Xsna algorithm. X and S are double precision matrices loaded from file.\n");
    printf("\tX: is an f x o matrix.\n");
    printf("\tB: is an f x n matrix.\n");
    printf("\titer: number of iterations.\n");
	printf("\titer: number of repetitions.\n");
    printf("Usage: %s X B <iter> <repe>\n", argv[0]);
    return -1;
  }

    inX = argv[1];
    inB = argv[2];
    iters= atoi(argv[3]);
    repe= atoi(argv[4]);
  

   read_file_header(inX, &f, &o);
   read_file_header(inB, &f, &n);


  #ifdef With_MKL
    X = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
    Xr = (double *)mkl_malloc(f * o * sizeof(double), WRDLEN);
    B = (double *)mkl_malloc(f * n * sizeof(double), WRDLEN);
    W = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
    B0 = (double *)mkl_malloc(f * n * sizeof(double), WRDLEN);
    W0 = (double *)mkl_malloc(n * o * sizeof(double), WRDLEN);
    w0= (double *)mkl_malloc(f * sizeof(double), WRDLEN);
  #else
    X = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
    Xr = (double *)_mm_malloc(f * o * sizeof(double), WRDLEN);
    B = (double *)_mm_malloc(f * n * sizeof(double), WRDLEN);
    W = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
    B0 = (double *)_mm_malloc(f * n * sizeof(double), WRDLEN);
    W0 = (double *)_mm_malloc(n * o * sizeof(double), WRDLEN);
    w0 = (double *)_mm_malloc(f * sizeof(double), WRDLEN);
  #endif
 
  dread_file(inX, f, o,X);
  dread_file(inB, f, n,B);
  
 printf("Entering asna\n");
  
  for(i = 0; i < repe; i++)
  {
	//t =  dsecnd();
	t = omp_get_wtime();
	status = dasnaPar_cpu(f, n, o, X, B, W, iters,nnz);
	Time += omp_get_wtime() - t;
	//Time += dsecnd() - t;
  }
	Time = Time/repe;
  if (status != 0)
  {
    printf("Problems with asna. Error code: %d\n", status);
    return -1;
  }
   
  error=derrorAsna(f,o,n,X,B,W);
  dwrite_file("resultW.dat",n,o,W);
      
  printf("#  M    N     K      Iteration    Time (sec)      Error\n");
  printf("# --------------------------------------------------------------------\n");     
  printf("%4d  %4d  %4d    %4d        %1.5E    %1.8E\n", f, o, n, iters, Time, error);


  #ifdef With_MKL
    mkl_free(X);
    mkl_free(B);
    mkl_free(W);
    mkl_free(B0);
    mkl_free(W0);
  #else
    _mm_free(X);
    _mm_free(B);
    _mm_free(W);
    _mm_free(B0);
    _mm_free(W0);
  #endif

  return 0;
}

int dread_file(char *path, int rows, int cols, double *M) 
{
  FILE *fp;
  int tmp_rows = 0, 
      tmp_cols = 0;

  if ((fp = fopen(path, "r")) == NULL)
     return -1;

  fread(&tmp_rows, sizeof(int), 1, fp);
  fread(&tmp_cols, sizeof(int), 1, fp);

  if ((tmp_rows == rows) && (tmp_cols == cols))
  {
     fread(M, sizeof(double), rows * cols , fp);
     fclose(fp);
     return 0;
  }
  else 
  {
     printf("Inconsistency between arguments and headers\n");
     fclose(fp);
     return -1;
  }
}


int dwrite_file(char *path, int rows, int cols, double *M) 
{
  FILE *fp;

  if ((fp = fopen(path, "w")) == NULL)
     return -1;

  fwrite(&rows, sizeof(int), 1, fp);
  fwrite(&cols, sizeof(int), 1, fp);
  fwrite(M, sizeof(double), rows * cols , fp);

  fclose(fp);
  return 0;
}

int read_file_header(char *path, int *rows, int *cols) 
{
  FILE *fp;

  if ((fp = fopen(path, "r")) == NULL)
     return -1;

  fread(rows, sizeof(int), 1, fp);
  fread(cols, sizeof(int), 1, fp);

  fclose(fp);
  return 0;
}
