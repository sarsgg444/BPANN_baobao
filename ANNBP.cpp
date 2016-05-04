#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Array_Init.h"
#include "Array_Compute.h"

void ANNBP_train()
{

}

int ANNBP_FixWeight
(   ARRAY_2D *ann_LayerA_output,
    ARRAY_2D *ann_LayerB_error,
    ARRAY_2D *ann_LayerAB_weightDelta,
    double learnSpeed
){
    ARRAY_2D ann_LayerA_output_TRANS;
        ann_LayerA_output_TRANS.col = ann_LayerA_output->row;
        ann_LayerA_output_TRANS.row = ann_LayerA_output->col;
    malloc2d_double( &ann_LayerA_output_TRANS );

    /*--- 計算 DeltaWeight ---*/
    Array2dTranspose_double( ann_LayerA_output, &ann_LayerA_output_TRANS );
    Array2dMultiply_double( &ann_LayerA_output_TRANS, ann_LayerB_error, ann_LayerAB_weightDelta );
    Array2dEnlarge_double( ann_LayerAB_weightDelta, learnSpeed, ann_LayerAB_weightDelta);

    return 1;
}

int ANNBP_LayerA_ErrorValue											/*--- ANNBP_LayerA_ErrorValue 說明 ---*/
(   ARRAY_2D *ann_LayerA_output,			// ann_LayerA_output A層的輸出值
    ARRAY_2D *ann_LayerAB_weight,			// ann_LayerAB_weight AB層之間的權重
    ARRAY_2D *ann_LayerB_error,		    	// ann_LayerB_error B層的誤差項(已知，A的右邊(後面)一層)
    ARRAY_2D *ann_LayerA_error	        	// ann_LayerA_error A層的誤差項(未知，要計算求得)
){
    int col;
    ARRAY_2D ann_LayerB_error_TRANS;
        ann_LayerB_error_TRANS.row = ann_LayerB_error->col;
        ann_LayerB_error_TRANS.col = ann_LayerB_error->row;
    malloc2d_double( &ann_LayerB_error_TRANS );
    ARRAY_2D temp;
        temp.row = ann_LayerAB_weight->row;
        temp.col = ann_LayerB_error->col;
        malloc2d_double( &temp );

	/*--- 檢查矩陣大小是否正確 ---*/
        if( ann_LayerA_output->row != ann_LayerA_error->row || ann_LayerA_output->col != ann_LayerA_error->col )
        {
            printf("Error!! Size of ann_LayerA_output != Size of ann_LayerA_error.\n");
            return 0;
        }
        if( ann_LayerAB_weight->col != ann_LayerB_error->col)
        {
            printf("Error!! rows of ann_LayerAB_weight != rows of ann_LayerB_error.\n");
            return 0;
        }
        if( ann_LayerA_error->col != ann_LayerAB_weight->row )
        {
            printf("Error!! cols of ann_LayerA_error != rows of ann_LayerAB_weight.\n");
            return 0;
        }

 	/*--- 計算LayerA誤差項 ---*/
        Array2dTranspose_double( ann_LayerB_error, &ann_LayerB_error_TRANS ); // 轉置 LayerA
        Array2dMultiply_double( ann_LayerAB_weight, &ann_LayerB_error_TRANS, &temp );
        for( col = 0; col<ann_LayerA_error->col; col++ )
        {
            ann_LayerA_error->arr[0][col] = ann_LayerA_output->arr[0][col] * ( 1-ann_LayerA_output->arr[0][col] ) * temp.arr[col][0];
        }

    return 1;
}

int ActivationFunction( ARRAY_2D *arrayA, ARRAY_2D *arrayOut )
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

    /*--- 激活函式 ---*/
        for( row = 0; row < arrayA->row; row++)
        {
            for( col = 0; col < arrayA->col; col++)
            {
                arrayOut->arr[row][col] = 1 / ( 1 + exp(-1*arrayA->arr[row][col]) );
            }
        }
    return 1;
}

char LoadTrainData( FILE* trainData, ARRAY_2D *ann_LayerI_input, ARRAY_2D *ann_LayerT_output )
{
    int i;
    char check;

    for( i=0; i<ann_LayerI_input->col; i++ )
        fscanf( trainData, "%lf,", &ann_LayerI_input->arr[0][i] );
    for( i=0; i<ann_LayerT_output->col; i++ )
        fscanf( trainData, "%lf,", &ann_LayerT_output->arr[0][i] );

    check = fgetc( trainData );

    return check;
}

int SaveAnn
(    FILE* SaveData,
     ARRAY_2D *ann_LayerIH_weight,
     ARRAY_2D *ann_LayerH_threshold,
     ARRAY_2D *ann_LayerHO_weight,
     ARRAY_2D *ann_LayerO_threshold
){
    int i, j;

    /*--- 儲存 ann_LayerIH_weight ---*/
    fprintf( SaveData, "ann_LayerIH_weight\n");
    for( i=0; i<ann_LayerIH_weight->row; i++ ){
        for( j=0; j<ann_LayerIH_weight->col; j++ ){
            fprintf( SaveData, "%.2lf,", ann_LayerIH_weight->arr[i][j] );
        }
        fprintf( SaveData, "\n" );
    }

    /*--- 儲存 ann_LayerH_threshold ---*/
    fprintf( SaveData, "ann_LayerH_threshold\n");
    for( i=0; i<ann_LayerH_threshold->row; i++ ){
        for( j=0; j<ann_LayerH_threshold->col; j++ ){
            fprintf( SaveData, "%.2lf,", ann_LayerH_threshold->arr[i][j] );
        }
        fprintf( SaveData, "\n" );
    }

    /*--- 儲存 ann_LayerHO_weight ---*/
    fprintf( SaveData, "ann_LayerHO_weight\n");
    for( i=0; i<ann_LayerHO_weight->row; i++ ){
        for( j=0; j<ann_LayerHO_weight->col; j++ ){
            fprintf( SaveData, "%.2lf,", ann_LayerHO_weight->arr[i][j] );
        }
        fprintf( SaveData, "\n" );
    }

    /*--- 儲存 ann_LayerO_threshold ---*/
    fprintf( SaveData, "ann_LayerO_threshold\n");
    for( i=0; i<ann_LayerO_threshold->row; i++ ){
        for( j=0; j<ann_LayerO_threshold->col; j++ ){
            fprintf( SaveData, "%.2lf,", ann_LayerO_threshold->arr[i][j] );
        }
        fprintf( SaveData, "\n" );
    }

    return 1;
}

int LoadAnn
(    FILE* LoadData,
     ARRAY_2D *ann_LayerIH_weight,
     ARRAY_2D *ann_LayerH_threshold,
     ARRAY_2D *ann_LayerHO_weight,
     ARRAY_2D *ann_LayerO_threshold
){
    int i, j;
    char datastring[256];

    /*--- 儲存 ann_LayerIH_weight ---*/
    fscanf( LoadData, "%s", datastring );
    for( i=0; i<ann_LayerIH_weight->row; i++ ){
        for( j=0; j<ann_LayerIH_weight->col; j++ ){
            fscanf( LoadData, "%lf,", &ann_LayerIH_weight->arr[i][j] );
        }
        fgetc( LoadData );
    }

    /*--- 儲存 ann_LayerH_threshold ---*/
    fscanf( LoadData, "%s", datastring );
    for( i=0; i<ann_LayerH_threshold->row; i++ ){
        for( j=0; j<ann_LayerH_threshold->col; j++ ){
            fscanf( LoadData, "%lf,", &ann_LayerH_threshold->arr[i][j] );
        }
        fgetc( LoadData );
    }

    /*--- 儲存 ann_LayerHO_weight ---*/
    fscanf( LoadData, "%s", datastring );
    for( i=0; i<ann_LayerHO_weight->row; i++ ){
        for( j=0; j<ann_LayerHO_weight->col; j++ ){
            fscanf( LoadData, "%lf,", &ann_LayerHO_weight->arr[i][j] );
        }
        fgetc( LoadData );
    }

    /*--- 儲存 ann_LayerO_threshold ---*/
    fscanf( LoadData, "%s", datastring );
    for( i=0; i<ann_LayerO_threshold->row; i++ ){
        for( j=0; j<ann_LayerO_threshold->col; j++ ){
            fscanf( LoadData, "%lf,", &ann_LayerO_threshold->arr[i][j] );
        }
        fgetc( LoadData );
    }

    return 1;
}
