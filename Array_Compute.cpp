/*-     Version:1.0         -*/
/*-     Date:2016/4/15      -*/
/*-     Author:Kabi         -*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Array_Compute.h"
#include "Array_Init.h"
#include "Array_Struct.h"

int Array2dRandom_double( ARRAY_2D *buffer, double rangeMax, double rangeMin )
{
    int i, j;
    double range;
    double **grid = buffer->arr;
    int nrows = buffer->row;
    int ncols = buffer->col;

    //檢查亂數範圍
    range = rangeMax - rangeMin;
    if( range <= 0 )
    {
        printf("Error random range.\n");
        return 0;
    }

    for( i=0; i<nrows; i++ )
    {
        for( j=0; j<ncols; j++ )
        {
            grid[i][j] = rangeMin + fmod( (double)rand()/100, range );
        }
    }

    return 1;
}

void Array2dPrintf_double( ARRAY_2D buffer )
{
    int i, j;
    double **grid;
    int nrows;
    int ncols;
    grid = buffer.arr;
    nrows = buffer.row;
    ncols = buffer.col;

    for( i=0; i<nrows; i++ )
    {
        for( j=0; j<ncols; j++ )
        {
            printf(" %lf ", grid[i][j] );
        }
        printf("\n");
    }
}


int Array2dMultiply_double( ARRAY_2D *arrayA, ARRAY_2D *arrayB, ARRAY_2D *arrayOut )
{
    int rowA, colA;
    int rowB, colB;

    Array2dInit_double( arrayOut->arr, arrayOut->row, arrayOut->col );   //輸出矩陣歸零

    /*--- 檢查矩陣大小是否可以相乘 ---*/
        if( arrayA->col != arrayB->row )
        {
            printf("Error!! cols of array_1 != rows of array_2.\n");
            return 0;
        }
        if( arrayA->row != arrayOut->row )
        {
            printf("Error!! rows of array_1 != rows of array_Out.\n");
            return 0;
        }
        if( arrayB->col != arrayOut->col )
        {
            printf("Error!! cols of array_2 != cols of array_Out.\n");
            return 0;
        }


    /*--- 矩陣相乘 ---*/
        for( rowA = 0; rowA < arrayA->row; rowA++)
        {
            for( colB = 0; colB < arrayB->col; colB++)
            {
                for( colA = 0; colA < arrayA->col; colA++)
                {
                    rowB = colA;
                    arrayOut->arr[rowA][colB] = arrayOut->arr[rowA][colB] + arrayA->arr[rowA][colA] * arrayB->arr[rowB][colB];
                }
            }
        }

    return 1;
}

int Array2dAdd_double( ARRAY_2D *arrayA, ARRAY_2D *arrayB, ARRAY_2D *arrayOut )
{
    int row, col;

    /*--- 檢查矩陣大小是否一樣 ---*/
        if( arrayA->col != arrayB->col || arrayB->col != arrayOut->col )
        {
            printf("Error!! cols of array_1 != rows of array_2 != cols of array_Out.\n");
            return 0;
        }
        if( arrayA->row != arrayB->row || arrayB->row != arrayOut->row )
        {
            printf("Error!! rows of array_1 != rows of array_2 != rows of array_Out.\n");
            return 0;
        }


    /*--- 矩陣相加 ---*/
        for( row = 0; row < arrayA->row; row++)
        {
            for( col = 0; col < arrayA->col; col++)
            {
                arrayOut->arr[row][col] = arrayA->arr[row][col] + arrayB->arr[row][col];
            }
        }

    return 1;
}

int Array2dSub_double( ARRAY_2D *arrayA, ARRAY_2D *arrayB, ARRAY_2D *arrayOut )
{
    int row, col;

    /*--- 檢查矩陣大小是否一樣 ---*/
        if( arrayA->col != arrayB->col || arrayB->col != arrayOut->col )
        {
            printf("Error!! cols of array_1 != rows of array_2 != cols of array_Out.\n");
            return 0;
        }
        if( arrayA->row != arrayB->row || arrayB->row != arrayOut->row )
        {
            printf("Error!! rows of array_1 != rows of array_2 != rows of array_Out.\n");
            return 0;
        }


    /*--- 矩陣相加 ---*/
        for( row = 0; row < arrayA->row; row++)
        {
            for( col = 0; col < arrayA->col; col++)
            {
                arrayOut->arr[row][col] = arrayA->arr[row][col] - arrayB->arr[row][col];
            }
        }

    return 1;
}


int Array2dEnlarge_double( ARRAY_2D *arrayA, double enlargeNum, ARRAY_2D *arrayOut )
{
    int row, col;

    /*--- 檢查矩陣大小是否一樣 ---*/
        if( arrayA->col != arrayOut->col )
        {
            printf("Error!! cols of array_1 != cols of array_Out.\n");
            return 0;
        }
        if( arrayA->row != arrayOut->row )
        {
            printf("Error!! rows of array_1 != rows of array_Out.\n");
            return 0;
        }


    /*--- 矩陣相加 ---*/
        for( row = 0; row < arrayA->row; row++)
        {
            for( col = 0; col < arrayA->col; col++)
            {
                arrayOut->arr[row][col] = arrayA->arr[row][col] * enlargeNum;
            }
        }

    return 1;
}

int Array2dTranspose_double( ARRAY_2D *buffer, ARRAY_2D *buffer2 )
{
    int row, col;
    double **arrayA = buffer->arr;
    int rowsA = buffer->row;
    int colsA = buffer->col;
    double **arrayOut = buffer2->arr;
    int rowsOut = buffer2->row;
    int colsOut = buffer2->col;



    /*--- 檢查矩陣大小是否一樣 ---*/
        if( rowsA != colsOut )
        {
            printf("Error!! rows of arrayA != cols of colsOut.\n");
            return 0;
        }
        if( colsA != rowsOut )
        {
            printf("Error!! cols of arrayA != rows of array_Out.\n");
            return 0;
        }


    /*--- 矩陣轉置 ---*/
        for( row = 0; row < rowsA; row++)
        {
            for( col = 0; col < colsA; col++)
            {
                arrayOut[col][row] = arrayA[row][col];
            }
        }

    return 1;
}
//double Array1dMultiply_double( double *arrayIn1, double *arrayIn2, int arraySize )
//{
//    int i;
//    double result = 0;
//
//    /*--- 矩陣相乘 ---*/
//    for (i=0; i<arraySize; i++)
//    {
//        result = result + arrayIn1[i] * arrayIn2[i];
//    }
//
//    return result;
//}
