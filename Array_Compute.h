#ifndef H_ARRAY_COMPUTE
#define H_ARRAY_COMPUTE

#include "Array_Struct.h"

int Array2dRandom_double( ARRAY_2D *buffer, double rangeMax, double rangeMin );
void Array2dPrintf_double( ARRAY_2D buffer );
int Array2dTranspose_double( ARRAY_2D *buffer, ARRAY_2D *buffer2 );

int Array2dMultiply_double( ARRAY_2D *arrayA, ARRAY_2D *arrayB, ARRAY_2D *arrayOut );
int Array2dAdd_double( ARRAY_2D *arrayA, ARRAY_2D *arrayB, ARRAY_2D *arrayOut );
int Array2dSub_double( ARRAY_2D *arrayA, ARRAY_2D *arrayB, ARRAY_2D *arrayOut );
int Array2dEnlarge_double( ARRAY_2D *arrayA, double enlargeNum, ARRAY_2D *arrayOut );



#endif // H_ARRAY_COMPUTE
