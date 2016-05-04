#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ANNBP.h"
#include "Array_Init.h"
#include "Array_Compute.h"
#include "Array_Struct.h"

//迭代次數限制
#define ITERATION_MAX 10000
//每層節點數
#define N 11    //LayerI
#define M 10    //LayerH
#define L 1    //LayerO
//亂數上下限
#define RAND_W_MAX 1.0     //亂數權重的最大值
#define RAND_W_MIN -1.0    //亂數權重的最小值
#define RAND_THETA_MAX 10   //亂數門檻的最大值
#define RAND_THETA_MIN -10  //亂數門檻的最小值

//                                                                                                  //
//    變數格式:前綴字_哪一層_何種資料                                                               //
//        --前綴字:ann                                                                              //
//        --哪一層:LayerI, LayerH, LayerO, LayerIH, LayerHO, LayerT                                 //
//        --何種資料:threshold, input, output, weight, error, thresholdDelta, weightDelta           //
//                                                                                                  //



int main()
{

    /*--- Variable ---*/
    //設定值
        int iteration = 0;        //迭代次數
        int dataCounter;
        double learnSpeed = 0.5;    //學習速率
    //ANN 矩陣
        //輸入資料、各節點門檻、節點間權重、目標輸出

        ARRAY_2D ann_LayerI_input;
            ann_LayerI_input.row = 1;
            ann_LayerI_input.col = N;
        ARRAY_2D ann_LayerIH_weight;
            ann_LayerIH_weight.row = N;
            ann_LayerIH_weight.col = M;
        ARRAY_2D ann_LayerH_threshold;
            ann_LayerH_threshold.row = 1;
            ann_LayerH_threshold.col = M;
        ARRAY_2D ann_LayerHO_weight;
            ann_LayerHO_weight.row = M;
            ann_LayerHO_weight.col = L;
        ARRAY_2D ann_LayerO_threshold;
            ann_LayerO_threshold.row = 1;
            ann_LayerO_threshold.col = L;
        ARRAY_2D ann_LayerT_output;
            ann_LayerT_output.row = 1;
            ann_LayerT_output.col = L;
        ARRAY_2D ann_LayerOT_weight;
            ann_LayerOT_weight.row = L;
            ann_LayerOT_weight.col = L;
        //各節點輸出、誤差值
        ARRAY_2D ann_LayerH_output;
            ann_LayerH_output.row = 1;
            ann_LayerH_output.col = M;
        ARRAY_2D ann_LayerH_error;
            ann_LayerH_error.row = 1;
            ann_LayerH_error.col = M;
        ARRAY_2D ann_LayerO_output;
            ann_LayerO_output.row = 1;
            ann_LayerO_output.col = L;
        ARRAY_2D ann_LayerO_error;
            ann_LayerO_error.row = 1;
            ann_LayerO_error.col = L;
        ARRAY_2D ann_LayerOT_error;
            ann_LayerOT_error.row = 1;
            ann_LayerOT_error.col = L;
        //各節點門檻、節點間權重的誤差
        ARRAY_2D ann_LayerIH_weightDelta;
            ann_LayerIH_weightDelta.row = N;
            ann_LayerIH_weightDelta.col = M;
        ARRAY_2D ann_LayerH_thresholdDelta;
            ann_LayerH_thresholdDelta.row = 1;
            ann_LayerH_thresholdDelta.col = M;
        ARRAY_2D ann_LayerHO_weightDelta;
            ann_LayerHO_weightDelta.row = M;
            ann_LayerHO_weightDelta.col = L;
        ARRAY_2D ann_LayerO_thresholdDelta;
            ann_LayerO_thresholdDelta.row = 1;
            ann_LayerO_thresholdDelta.col = L;
    //時間亂數seed
        srand(time(NULL));
    //檔案
        FILE* trainData = fopen("trainData.txt","r");
            if( trainData == 0 ) return 0; // 訓練資料
        FILE* WeightAndThresholdRecord; // 訓練紀錄
        char filenameSave[256];
        int fileSwitch;
    //計時器
        clock_t  t1, t2;

    /*--- Init Array ---*/
    //動態建立陣列
        //建立"輸入資料、各節點門檻、節點間權重、目標輸出"的陣列
        malloc2d_double( &ann_LayerI_input );
        malloc2d_double( &ann_LayerIH_weight );
        malloc2d_double( &ann_LayerH_threshold );
        malloc2d_double( &ann_LayerHO_weight );
        malloc2d_double( &ann_LayerO_threshold );
        malloc2d_double( &ann_LayerT_output );
        malloc2d_double( &ann_LayerOT_weight );
        //建立"各節點輸出、誤差值"的陣列
        malloc2d_double( &ann_LayerH_output );
        malloc2d_double( &ann_LayerH_error );
        malloc2d_double( &ann_LayerO_output );
        malloc2d_double( &ann_LayerO_error );
        malloc2d_double( &ann_LayerOT_error );
        //建立"各節點門檻、節點間權重的誤差"的陣列
        malloc2d_double( &ann_LayerIH_weightDelta );
        malloc2d_double( &ann_LayerH_thresholdDelta );
        malloc2d_double( &ann_LayerHO_weightDelta );
        malloc2d_double( &ann_LayerO_thresholdDelta );

    //輸入初始值(權重、門檻)
        printf("亂數門檻、權重請按-1\n讀檔請按檔案編號\n");
        scanf("%d",&fileSwitch);
        if( fileSwitch == -1 )
        {
            Array2dRandom_double( &ann_LayerIH_weight, RAND_W_MAX, RAND_W_MIN );
            Array2dRandom_double( &ann_LayerH_threshold, RAND_THETA_MAX, RAND_THETA_MIN );
            Array2dRandom_double( &ann_LayerHO_weight, RAND_W_MAX, RAND_W_MIN );
            Array2dRandom_double( &ann_LayerO_threshold, RAND_THETA_MAX, RAND_THETA_MIN );
        }
        else
        {
            sprintf( filenameSave, "WeightAndThresholdRecord_%d.txt", fileSwitch );
            WeightAndThresholdRecord = fopen(filenameSave,"r"); // 訓練紀錄
            LoadAnn( WeightAndThresholdRecord, &ann_LayerIH_weight, &ann_LayerH_threshold, &ann_LayerHO_weight, &ann_LayerO_threshold );
            fclose(WeightAndThresholdRecord);
            /*--- Debug ---*/
                Array2dPrintf_double( ann_LayerIH_weight );
                Array2dPrintf_double( ann_LayerH_threshold );
                Array2dPrintf_double( ann_LayerHO_weight );
                Array2dPrintf_double( ann_LayerO_threshold );
        }

    /*--- 類神經訓練(所有訓練資料跑 ITERATION_MAX 次) ---*/
        while( iteration < ITERATION_MAX )
        {
                fseek( trainData, 0, SEEK_SET );    // 回到第一筆訓練資料
                dataCounter = 0;
            /*--- 紀錄權重、門檻 ---*/
                sprintf( filenameSave, "WeightAndThresholdRecord_%d.txt", iteration );
                WeightAndThresholdRecord = fopen(filenameSave,"w"); // 訓練紀錄
                SaveAnn( WeightAndThresholdRecord, &ann_LayerIH_weight, &ann_LayerH_threshold, &ann_LayerHO_weight, &ann_LayerO_threshold );
                fclose(WeightAndThresholdRecord);
            /*--- 類神經訓練(所有訓練資料) ---*/
                while( LoadTrainData( trainData, &ann_LayerI_input, &ann_LayerT_output ) != EOF ) // 逐行讀取訓練資料
                {
                    /*--- Debug ---*/
//                        printf("input data %d \n", dataCounter );
//                        Array2dPrintf_double( ann_LayerI_input, 1, N );
//                        printf("Targetoutput %d \n", dataCounter );
//                        Array2dPrintf_double( ann_LayerT_output, 1, L );
                          t1=clock();

                    /*--- 輸入層到隱藏層 ---*/
                        // X*W_ih;
                        // 輸入資料*權重
                        Array2dMultiply_double( &ann_LayerI_input, &ann_LayerIH_weight, &ann_LayerH_output );
printf("ann_LayerH_input = \n");
Array2dPrintf_double( ann_LayerH_output );
                        // ( X*W_ih )+ Theta_h;
                        // (輸入資料*權重_ih)+門檻值_h
                        Array2dAdd_double( &ann_LayerH_threshold, &ann_LayerH_output, &ann_LayerH_output );
printf("ann_LayerH_input+theta = \n");
Array2dPrintf_double( ann_LayerH_output );
                        // 將 { (輸入資料*權重_ih)+門檻值_h } 輸入到 ActivationFunction 計算該節點的輸出
                        ActivationFunction( &ann_LayerH_output, &ann_LayerH_output );
printf("ann_LayerH_output = \n");
Array2dPrintf_double( ann_LayerH_output );

                    /*--- 隱藏層到輸出層 ---*/
                        // O_h * W_ho;
                        // 隱藏層輸出*權重
                        Array2dMultiply_double( &ann_LayerH_output, &ann_LayerHO_weight, &ann_LayerO_output );
printf("ann_LayerO_input = \n");
Array2dPrintf_double( ann_LayerO_output );
                        // ( O_h * W_ho )+ Theta_o;
                        // (隱藏層輸出*權重)+門檻值_o
                        Array2dAdd_double( &ann_LayerO_threshold, &ann_LayerO_output, &ann_LayerO_output );
printf("ann_LayerO_input+theta = \n");
Array2dPrintf_double( ann_LayerO_output );
                        // 將 { (隱藏層輸出*權重)+門檻值_o } 輸入到 ActivationFunction 計算該節點的輸出
                        ActivationFunction( &ann_LayerO_output, &ann_LayerO_output );
printf("ann_LayerO_input = \n");
Array2dPrintf_double( ann_LayerO_output );

                    /*--- 計算誤差項 ---*/
                        //計算輸出層誤差項
                        UnitArray2dInit_double( &ann_LayerOT_weight );
                        Array2dSub_double( &ann_LayerT_output, &ann_LayerO_output, &ann_LayerOT_error );
                        ANNBP_LayerA_ErrorValue( &ann_LayerO_output, &ann_LayerOT_weight, &ann_LayerOT_error, &ann_LayerO_error );
printf("ann_LayerO_error = \n");
Array2dPrintf_double( ann_LayerO_error );
                        //計算隱藏層誤差項
                        ANNBP_LayerA_ErrorValue( &ann_LayerH_output, &ann_LayerHO_weight, &ann_LayerO_error, &ann_LayerH_error );
printf("ann_LayerH_error = \n");
Array2dPrintf_double( ann_LayerH_error );

                    /*--- 修正門檻 ---*/
                        //修正輸出層門檻
                        Array2dEnlarge_double(&ann_LayerO_error, learnSpeed, &ann_LayerO_thresholdDelta );
                        Array2dAdd_double( &ann_LayerO_threshold, &ann_LayerO_thresholdDelta, &ann_LayerO_threshold );
printf("ann_LayerO_thresholdDelta = \n");
Array2dPrintf_double( ann_LayerO_thresholdDelta );
printf("NEW ann_LayerO_thresholdDelta = \n");
Array2dPrintf_double( ann_LayerO_threshold );
                        //修正隱藏層門檻
                        Array2dEnlarge_double(&ann_LayerH_error, learnSpeed, &ann_LayerH_thresholdDelta );
                        Array2dAdd_double( &ann_LayerH_threshold, &ann_LayerH_thresholdDelta, &ann_LayerH_threshold );
printf("ann_LayerH_thresholdDelta = ");
Array2dPrintf_double( ann_LayerH_thresholdDelta );
printf("NEW ann_LayerH_thresholdDelta = ");
Array2dPrintf_double( ann_LayerH_threshold );

                    /*--- 修正權重 ---*/
                        //修正隱藏-輸出層權重
                        ANNBP_FixWeight( &ann_LayerH_output, &ann_LayerO_error, &ann_LayerHO_weightDelta, learnSpeed );
                        Array2dAdd_double( &ann_LayerHO_weight, &ann_LayerHO_weightDelta, &ann_LayerHO_weight );
printf("ann_LayerHO_weightDelta = \n");
Array2dPrintf_double( ann_LayerHO_weightDelta );
printf("NEW ann_LayerHO_weight = \n");
Array2dPrintf_double( ann_LayerHO_weight );
                        //修正輸入-隱藏層權重
                        ANNBP_FixWeight( &ann_LayerI_input, &ann_LayerH_error, &ann_LayerIH_weightDelta, learnSpeed );
                        Array2dAdd_double( &ann_LayerIH_weight, &ann_LayerIH_weightDelta, &ann_LayerIH_weight );
printf("ann_LayerIH_weightDelta = \n");
Array2dPrintf_double( ann_LayerIH_weightDelta );
printf("NEW ann_LayerIH_weight = \n");
Array2dPrintf_double( ann_LayerIH_weight );

                    /*--- Debug ---*/
                        printf("data %d\n",dataCounter);
                        dataCounter++;

//                        printf("ann_LayerH_error = ");
//                        Array2dPrintf_double( ann_LayerH_error, 1, M );
//                        printf("ann_LayerO_error = ");
//                        Array2dPrintf_double( ann_LayerO_error, 1, L );

//                        printf("ANN Output = ");
//                        Array2dPrintf_double( &ann_LayerO_output );
//                        printf("Target Output = ");
//                        Array2dPrintf_double( &ann_LayerT_output );
//                        printf("----------\n");

                        t2=clock();
                        printf("%lf\n", (t2-t1)/(double)(CLOCKS_PER_SEC));

                    iteration++;    //迭代次數+1
                }
        }

    return 0;
}
