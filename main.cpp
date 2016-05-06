#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ANNBP.h"
#include "Array_Init.h"
#include "Array_Compute.h"
#include "Array_Struct.h"
#include "FileIO.h"

//迭代次數限制
#define ITERATION_MAX 10000

//                                                                                                  //
//    變數格式:前綴字_哪一層_何種資料                                                               //
//        --前綴字:ann                                                                              //
//        --哪一層:layerI, layerH, layerO, layerH[], LayerHO,                                       //

int main()
{
    /*--- Variable ---*/
    //設定值
        int iteration = 0;        //迭代次數
        int dataCounter = 0;
        double learnSpeed = 0.5;    //學習速率
        int i;
    //ANN 矩陣
        //網路架構，存放層數n及每層的節點數
            ARCHITECTURE network;
        //各層陣列資料
            LAYER ann_layerI;   //數量 1 層
            LAYER *ann_layerH;  //數量 n-2 層
            LAYER ann_layerO;   //數量 1 層
        //權重資料，存放層與層之間的權重資料，數量 n-1
            WEIGHT_LINE *ann_weight;
    //訓練資料
        TRAINING_DATA dataBase;
    //時間亂數seed
        srand(time(NULL));
    //檔案
        char filenameRestore[256];
        char filenameTrain[256];
            sprintf( filenameTrain, "abalone-5-1tra.dat" );
        int fileSwitch;
        FILE* WeightAndThresholdRecord; // 訓練紀錄
    //計時器
        clock_t  t1, t2;

    /*--- Init Array ---*/
        //讀取網路架構
            LoadNetworkArchitecture( "setting/node_of_layer.txt", &network );
        //配置各層動態記憶體
            Init_ann_layerI( &ann_layerI, network );
            Init_ann_layerH( &ann_layerH, network );
            Init_ann_layerO( &ann_layerO, network );
        //配置所有層與層之間的權重動態記憶體
            Init_ann_weight( &ann_weight, network );
        //讀取訓練資料，並配置存放用的動態記憶體
            dataBase.data_size = FileRow( filenameTrain );      //計算資料筆數
            if( !LoadTrainData( filenameTrain , &dataBase, network ) )
                return 0;

        //輸入初始值(權重、門檻)
            printf("亂數門檻、權重請按-1\n讀檔請按檔案編號\n");
            scanf("%d",&fileSwitch);
            if( fileSwitch == -1 )
                Random_ANN( ann_layerH, &ann_layerO, ann_weight, network );
            else{
                sprintf( filenameRestore, "WeightAndThresholdRecord_%d.txt", fileSwitch );
                LoadAnn( filenameRestore, ann_layerH, &ann_layerO, ann_weight, network );
            }


    /*--- 類神經訓練(所有訓練資料跑 ITERATION_MAX 次) ---*/
        while( iteration < ITERATION_MAX )
        {
            //放入一筆資料
                PutRowTrainData( &ann_layerI, dataBase, dataCounter );
            /*--- 順向預測 ---*/
                ///*--- 輸入層到隱藏層 ---*/
                    //權重線運算
                    ANNBP_LineCompute( ann_layerI, ann_weight[0], &ann_layerH[0] );
                ///*--- 隱藏層 ---*/
                    for( i=0; i<network.layer-3; i++ ){     // 3代表輸入、輸出跟最後面的隱藏層
                        ANNBP_LayerCompute( &ann_layerH[i] );
                        ANNBP_LineCompute( ann_layerH[i], ann_weight[i+1], &ann_layerH[i] );
                    }
                SaveAnn( "WeightAndThresholdRecord_-2.txt", ann_layerH, ann_layerO, ann_weight, network );
//                // X*W_ih;
//                // 輸入資料*權重
//                Array2dMultiply_double( &an, &ann_LayerIH_weight, &ann_LayerH_output );
//printf("ann_LayerH_input = \n");
//Array2dPrintf_double( ann_LayerH_output );
//                // ( X*W_ih )+ Theta_h;
//                // (輸入資料*權重_ih)+門檻值_h
//                Array2dAdd_double( &ann_LayerH_threshold, &ann_LayerH_output, &ann_LayerH_output );
//printf("ann_LayerH_input+theta = \n");
//Array2dPrintf_double( ann_LayerH_output );
//                // 將 { (輸入資料*權重_ih)+門檻值_h } 輸入到 ActivationFunction 計算該節點的輸出
//                ActivationFunction( &ann_LayerH_output, &ann_LayerH_output );
//printf("ann_LayerH_output = \n");
//Array2dPrintf_double( ann_LayerH_output );
        }


//        while( iteration < ITERATION_MAX )
//        {
//                fseek( trainData, 0, SEEK_SET );    // 回到第一筆訓練資料
//                dataCounter = 0;
//            /*--- 紀錄權重、門檻 ---*/
//                sprintf( filenameSave, "WeightAndThresholdRecord_%d.txt", iteration );
//                WeightAndThresholdRecord = fopen(filenameSave,"w"); // 訓練紀錄
//                SaveAnn( WeightAndThresholdRecord, &ann_LayerIH_weight, &ann_LayerH_threshold, &ann_LayerHO_weight, &ann_LayerO_threshold );
//                fclose(WeightAndThresholdRecord);
//            /*--- 類神經訓練(所有訓練資料) ---*/
//                while( LoadTrainData( trainData, &ann_LayerI_input, &ann_LayerT_output ) != EOF ) // 逐行讀取訓練資料
//                {
//                    /*--- Debug ---*/
////                        printf("input data %d \n", dataCounter );
////                        Array2dPrintf_double( ann_LayerI_input, 1, N );
////                        printf("Targetoutput %d \n", dataCounter );
////                        Array2dPrintf_double( ann_LayerT_output, 1, L );
//                          t1=clock();
//
//                    /*--- 輸入層到隱藏層 ---*/
//                        // X*W_ih;
//                        // 輸入資料*權重
//                        Array2dMultiply_double( &ann_LayerI_input, &ann_LayerIH_weight, &ann_LayerH_output );
//printf("ann_LayerH_input = \n");
//Array2dPrintf_double( ann_LayerH_output );
//                        // ( X*W_ih )+ Theta_h;
//                        // (輸入資料*權重_ih)+門檻值_h
//                        Array2dAdd_double( &ann_LayerH_threshold, &ann_LayerH_output, &ann_LayerH_output );
//printf("ann_LayerH_input+theta = \n");
//Array2dPrintf_double( ann_LayerH_output );
//                        // 將 { (輸入資料*權重_ih)+門檻值_h } 輸入到 ActivationFunction 計算該節點的輸出
//                        ActivationFunction( &ann_LayerH_output, &ann_LayerH_output );
//printf("ann_LayerH_output = \n");
//Array2dPrintf_double( ann_LayerH_output );
//
//                    /*--- 隱藏層到輸出層 ---*/
//                        // O_h * W_ho;
//                        // 隱藏層輸出*權重
//                        Array2dMultiply_double( &ann_LayerH_output, &ann_LayerHO_weight, &ann_LayerO_output );
//printf("ann_LayerO_input = \n");
//Array2dPrintf_double( ann_LayerO_output );
//                        // ( O_h * W_ho )+ Theta_o;
//                        // (隱藏層輸出*權重)+門檻值_o
//                        Array2dAdd_double( &ann_LayerO_threshold, &ann_LayerO_output, &ann_LayerO_output );
//printf("ann_LayerO_input+theta = \n");
//Array2dPrintf_double( ann_LayerO_output );
//                        // 將 { (隱藏層輸出*權重)+門檻值_o } 輸入到 ActivationFunction 計算該節點的輸出
//                        ActivationFunction( &ann_LayerO_output, &ann_LayerO_output );
//printf("ann_LayerO_input = \n");
//Array2dPrintf_double( ann_LayerO_output );
//
//                    /*--- 計算誤差項 ---*/
//                        //計算輸出層誤差項
//                        UnitArray2dInit_double( &ann_LayerOT_weight );
//                        Array2dSub_double( &ann_LayerT_output, &ann_LayerO_output, &ann_LayerOT_error );
//                        ANNBP_LayerA_ErrorValue( &ann_LayerO_output, &ann_LayerOT_weight, &ann_LayerOT_error, &ann_LayerO_error );
//printf("ann_LayerO_error = \n");
//Array2dPrintf_double( ann_LayerO_error );
//                        //計算隱藏層誤差項
//                        ANNBP_LayerA_ErrorValue( &ann_LayerH_output, &ann_LayerHO_weight, &ann_LayerO_error, &ann_LayerH_error );
//printf("ann_LayerH_error = \n");
//Array2dPrintf_double( ann_LayerH_error );
//
//                    /*--- 修正門檻 ---*/
//                        //修正輸出層門檻
//                        Array2dEnlarge_double(&ann_LayerO_error, learnSpeed, &ann_LayerO_thresholdDelta );
//                        Array2dAdd_double( &ann_LayerO_threshold, &ann_LayerO_thresholdDelta, &ann_LayerO_threshold );
//printf("ann_LayerO_thresholdDelta = \n");
//Array2dPrintf_double( ann_LayerO_thresholdDelta );
//printf("NEW ann_LayerO_thresholdDelta = \n");
//Array2dPrintf_double( ann_LayerO_threshold );
//                        //修正隱藏層門檻
//                        Array2dEnlarge_double(&ann_LayerH_error, learnSpeed, &ann_LayerH_thresholdDelta );
//                        Array2dAdd_double( &ann_LayerH_threshold, &ann_LayerH_thresholdDelta, &ann_LayerH_threshold );
//printf("ann_LayerH_thresholdDelta = ");
//Array2dPrintf_double( ann_LayerH_thresholdDelta );
//printf("NEW ann_LayerH_thresholdDelta = ");
//Array2dPrintf_double( ann_LayerH_threshold );
//
//                    /*--- 修正權重 ---*/
//                        //修正隱藏-輸出層權重
//                        ANNBP_FixWeight( &ann_LayerH_output, &ann_LayerO_error, &ann_LayerHO_weightDelta, learnSpeed );
//                        Array2dAdd_double( &ann_LayerHO_weight, &ann_LayerHO_weightDelta, &ann_LayerHO_weight );
//printf("ann_LayerHO_weightDelta = \n");
//Array2dPrintf_double( ann_LayerHO_weightDelta );
//printf("NEW ann_LayerHO_weight = \n");
//Array2dPrintf_double( ann_LayerHO_weight );
//                        //修正輸入-隱藏層權重
//                        ANNBP_FixWeight( &ann_LayerI_input, &ann_LayerH_error, &ann_LayerIH_weightDelta, learnSpeed );
//                        Array2dAdd_double( &ann_LayerIH_weight, &ann_LayerIH_weightDelta, &ann_LayerIH_weight );
//printf("ann_LayerIH_weightDelta = \n");
//Array2dPrintf_double( ann_LayerIH_weightDelta );
//printf("NEW ann_LayerIH_weight = \n");
//Array2dPrintf_double( ann_LayerIH_weight );
//
//                    /*--- Debug ---*/
//                        printf("data %d\n",dataCounter);
//                        dataCounter++;
//
////                        printf("ann_LayerH_error = ");
////                        Array2dPrintf_double( ann_LayerH_error, 1, M );
////                        printf("ann_LayerO_error = ");
////                        Array2dPrintf_double( ann_LayerO_error, 1, L );
//
////                        printf("ANN Output = ");
////                        Array2dPrintf_double( &ann_LayerO_output );
////                        printf("Target Output = ");
////                        Array2dPrintf_double( &ann_LayerT_output );
////                        printf("----------\n");
//
//                        t2=clock();
//                        printf("%lf\n", (t2-t1)/(double)(CLOCKS_PER_SEC));
//
//                    iteration++;    //迭代次數+1
//                }
//        }

    return 0;
}
