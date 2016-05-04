/*-     Version:1.0         -*/
/*-     Date:2016/4/15      -*/
/*-     Author:Kabi         -*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Array_Init.h"
#include "Array_Struct.h"

void ArrayInit_int(int input[],int ArraySize)
{
    int i;
    for( i=0; i<ArraySize; i++ )
    {
        input[i] = 0;
    }
}

void ArrayInit_double(double input[],int ArraySize)
{
    int i;
    for( i=0; i<ArraySize; i++ )
    {
        input[i] = 0;
    }
}

void ArrayInitSPNum_double(double input[],int ArraySize, double SPNum )
{
    int i;
    for( i=0; i<ArraySize; i++ )
    {
        input[i] = SPNum;
    }
}

void Array2dInit_double(double **input, int nrows, int ncols)
{
    int i;
    for( i=0; i<nrows; i++ )
    {
        ArrayInit_double( &input[i][0], ncols );
    }
}

void Array2dInitSPNum_double(double **input, int nrows, int ncols, double SPNum)
{
    int i;
    for( i=0; i<nrows; i++ )
    {
        ArrayInitSPNum_double( &input[i][0], ncols, SPNum );
    }
}

int UnitArray2dInit_double(ARRAY_2D *buffer)
{
    int i, j;

    if( buffer->row != buffer->col ){
        printf("Input array is not a square matrix.\n");
        return 0;
    }

    for( i=0; i<buffer->row; i++ )
    {
        for( j=0; j<buffer->col; j++ )
        {
            if( i == j )
                buffer->arr[i][j] = 1;
            else
                buffer->arr[i][j] = 0;
        }
    }

    return 1;
}


int malloc2d_int(int *** grid, int nrows, int ncols)
{
    int i;
    *grid = (int**)malloc( sizeof(int *) * nrows);

    if (*grid == NULL){
        printf("ERROR: out of memory\n");
        return 1;
    }

    for (i=0;i<nrows;i++){
        (*grid)[i] = (int*)malloc( sizeof(int) * ncols);
        if ((*grid)[i] == NULL){
            printf("ERROR: out of memory\n");
            return 1;
        }
    }
    //printf("Dynamic Array Allocated!\n");
    return 0;
}

double malloc2d_double(ARRAY_2D *buffer)
{
    int i;
    double ***grid = &(buffer->arr);
    int nrows = buffer->row;
    int ncols = buffer->col;

    //分配動態 2 維陣列記憶體空間
    *grid = (double**)malloc( sizeof(double *) * nrows);

    if (*grid == NULL){
        printf("ERROR: out of memory\n");
        return 0;
    }

    for (i=0;i<nrows;i++){
        (*grid)[i] = (double*)malloc( sizeof(double) * ncols);
        if ((*grid)[i] == NULL){
            printf("ERROR: out of memory\n");
            return 0;
        }
    }
    //printf("Allocated!\n");
    return 1;
}
