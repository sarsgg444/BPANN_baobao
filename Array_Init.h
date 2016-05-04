#ifndef H_ARRAY_INIT
#define H_ARRAY_INIT

#include "Array_Struct.h"

void ArrayInit_int(int input[],int ArraySize);
void ArrayInit_double(double input[],int ArraySize);
void ArrayInitSPNum_double(double input[],int ArraySize, double SPNum );
void Array2dInitSPNum_double(double **input, int nrows, int ncols, double SPNum);

void Array2dInit_double(double **input, int nrows, int ncols);
int UnitArray2dInit_double(ARRAY_2D *buffer);

int malloc2d_int(int *** grid, int nrows, int ncols);

//double malloc2d_double(double *** grid, int nrows, int ncols);
double malloc2d_double(ARRAY_2D *buffer);


#endif // H_ARRAY_INIT
