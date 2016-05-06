#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Array_Init.h"
#include "Array_Compute.h"
#include "ANNBP.h"
#include "FileIO.h"

int PutRowTrainData( LAYER* ann_layerI, TRAINING_DATA ann_data, int loadRow )
{
    int i;
    int layerI_r = ann_layerI->output.row;
    int layerI_c = ann_layerI->output.col;

    for( i=0; i<layerI_c; i++ )
        ann_layerI->output.arr[0][i]  = ann_data.feature.arr[loadRow][i];

    return 1;
}

int ANNBP_LineCompute( LAYER ann_layerA, WEIGHT_LINE weight ,LAYER *ann_layerB )
{
    Array2dMultiply_double( &ann_layerA.output, &weight.weight, &ann_layerB->input );
    return 1;
}

int ANNBP_LayerCompute( LAYER *ann_layerA )
{
    int col = ann_layerA->input.col;
    int i;

    for( i=0; i<col; i++ ){
        ann_layerA->output.arr[0][i] = ann_layerA->input.arr[0][i] + ann_layerA->theta.arr[0][i];  //加上門檻值 theta
        ann_layerA->output.arr[0][i] = ActivationFunction( ann_layerA->output.arr[0][i], ann_layerA->activationNum );
    }

    return 1;
}

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
    malloc2d_double( &ann_LayerA_output_TRANS, ann_LayerA_output->col, ann_LayerA_output->row);

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
    malloc2d_double( &ann_LayerB_error_TRANS, ann_LayerB_error->col, ann_LayerB_error->row);
    ARRAY_2D temp;
        temp.row = ann_LayerAB_weight->row;
        temp.col = ann_LayerB_error->col;
        malloc2d_double( &temp, ann_LayerAB_weight->row, ann_LayerB_error->col );

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

double ActivationFunction( double x, int function_switch )      //激活函數
{
    double result;

    /*--- 激活函式 ---*/
        switch( function_switch )
        {
            case NON:     // NON
                result =  x;
                break;
            case SIGMOID:     // Sigmoid function
                result =  1 / ( 1 + exp(-x) );
                break;
            case STEP:     // Step function
                if( x >= 0 )
                    result = 1;
                else
                    result = 0;
                break;
            default:
                printf("Error!!No function_switch %d \n",function_switch);
                return 0;
                break;
        }
    return result;
}

int LoadRowdata( FILE* fp, int* buffer )        //讀取一列資料
{
    char dataDisplay[256];
    char temp[256];
    int i=0;
    char check;

    fscanf( fp, "%[^:]:", dataDisplay );    //將冒號前的字串讀入
    printf("%s:",dataDisplay);
    do
    {
        fscanf( fp, "%[^,0-9]s", temp );    //讀字串直到是,或著0-9
        fscanf( fp, "%[0-9 ]s", temp );     //讀取參數
        buffer[i] = atoi(temp);
        printf("%d ",buffer[i]);
        i++;
        check = fgetc(fp);  //檢查是否到該行結尾
    }while( check != '\n' && check != EOF );    //還沒讀完一行就繼續
    fscanf( fp, "\n" );
    printf("\n");

    return i;
}

int LoadNetworkArchitecture( char *filename, ARCHITECTURE *network )        //讀取網路架構
{
    FILE* fp_NodeData = fopen( filename, "r" );

    if( fp_NodeData == NULL)
    {
        printf( "No file %s\n", filename );
        return 0;
    }

    network->nodeH = (int*) malloc( sizeof(int) * MAX_HIDDEN_LAYER );      //配置隱藏層最大層數記憶體空間

    network->layer = 0;
    network->layer = network->layer + LoadRowdata( fp_NodeData, &network->nodeI );     //
    network->layer = network->layer + LoadRowdata( fp_NodeData, network->nodeH );     //
    network->layer = network->layer + LoadRowdata( fp_NodeData, &network->nodeO );     //

    if( network->layer > MAX_HIDDEN_LAYER+2 )   //檢查是否超過最多層數限制， 2 代表輸入、輸出層的數量
    {
        printf("Hidden layer is bigger than MAX_HIDDEN_LAYER");
        return 0;
    }

    fclose(fp_NodeData);
    return 1;
}

int Init_ann_layerI( LAYER* ann_layerI, ARCHITECTURE network )      //分配 layerI 的動態記憶體
{
    ann_layerI->node = network.nodeI;   //讀取輸入層節點

    /*--- 根據輸入層節點，配置動態陣列 ---*/
    if( !malloc2d_double( &ann_layerI->input, 1, network.nodeI ) )
        return 0;
    if( !malloc2d_double( &ann_layerI->output, 1, network.nodeI ) )
        return 0;
    if( !malloc2d_double( &ann_layerI->theta, 1, network.nodeI ) )
        return 0;
    if( !malloc2d_double( &ann_layerI->theta_delta, 1, network.nodeI ) )
        return 0;
    if( !malloc2d_double( &ann_layerI->error, 1, network.nodeI ) )
        return 0;

    ann_layerI->activationNum = NON;    //輸入層不需要激活函數

    return 1;
}

int Init_ann_layerO( LAYER* ann_layerO, ARCHITECTURE network )      //分配 layerO 的動態記憶體
{
    ann_layerO->node = network.nodeO;   //讀取輸入層節點

    /*--- 根據輸入層節點，配置動態陣列 ---*/
    if( !malloc2d_double( &ann_layerO->input, 1, network.nodeO ) )
        return 0;
    if( !malloc2d_double( &ann_layerO->output, 1, network.nodeO ) )
        return 0;
    if( !malloc2d_double( &ann_layerO->theta, 1, network.nodeO ) )
        return 0;
    if( !malloc2d_double( &ann_layerO->theta_delta, 1, network.nodeO ) )
        return 0;
    if( !malloc2d_double( &ann_layerO->error, 1, network.nodeO ) )
        return 0;

    ann_layerO->activationNum = SIGMOID;    //輸出層激活函數

    return 1;
}

int Init_ann_layerH( LAYER** ann_layerH, ARCHITECTURE network )     //分配 layerH 的動態記憶體
{
    int layer;

    *ann_layerH = (LAYER*)malloc( sizeof(LAYER)* (network.layer-2) );    //根據層數配置隱藏層的動態記憶體
    LAYER *buffer = *ann_layerH;

    for( layer = 0; layer < network.layer-2; layer++ )
    {
        buffer[layer].node = network.nodeH[layer];  //讀取輸入層節點

        /*--- 根據輸入層節點，配置動態陣列 ---*/
        if( !malloc2d_double( &buffer[layer].input, 1, network.nodeH[layer] ) )
            return 0;
        if( !malloc2d_double( &buffer[layer].output, 1, network.nodeH[layer] ) )
            return 0;
        if( !malloc2d_double( &buffer[layer].theta, 1, network.nodeH[layer] ) )
            return 0;
        if( !malloc2d_double( &buffer[layer].theta_delta, 1, network.nodeH[layer] ) )
            return 0;
        if( !malloc2d_double( &buffer[layer].error, 1, network.nodeH[layer] ) )
            return 0;

        buffer[layer].activationNum = SIGMOID;    //隱藏層激活函數
    }

    return 1;
}

int Init_ann_weight( WEIGHT_LINE** weight, ARCHITECTURE network )       //分配 weight 的動態記憶體
{
    WEIGHT_LINE *buffer;
    int i;

    *weight = (WEIGHT_LINE*)malloc( sizeof(WEIGHT_LINE) * (network.layer-1) );      //網路有 n 層，權重就有 n-1 層
    buffer = *weight;

    for( i=0; i<network.layer-1; i++ )
    {
        if( i == 0 )    //輸入層與隱藏層中間的權重層
        {
            if( !malloc2d_double( &buffer[i].weight, network.nodeI, network.nodeH[i] ) )        //權重陣列配置，大小 nodeI*nodeH[0]
                return 0;
            if( !malloc2d_double( &buffer[i].weight_delta, network.nodeI, network.nodeH[i] ) )     //權重修正陣列配置，大小 nodeI*nodeH[0]
                return 0;
        }
        else if( i == network.layer-2 )   //隱藏層與輸出層中間的權重層
        {
            if( !malloc2d_double( &buffer[i].weight, network.nodeH[i-1], network.nodeO ) )        //權重陣列配置，大小 nodeH[final]*nodeO
                return 0;
            if( !malloc2d_double( &buffer[i].weight_delta, network.nodeH[i-1], network.nodeO ) )      //權重修正陣列配置，大小 nodeH[final]*nodeO
                return 0;
        }
        else    //隱藏層之間的權重層
        {
            if( !malloc2d_double( &buffer[i].weight, network.nodeH[i-1], network.nodeH[i] ) )        //權重陣列配置，大小 nodeH[i-1]*nodeH[i]
                return 0;
            if( !malloc2d_double( &buffer[i].weight_delta, network.nodeH[i-1], network.nodeH[i]) )      //權重修正陣列配置，大小 nodeH[i-1]*nodeH[i]
                return 0;
        }
    }

    return 1;
}

int Random_ANN( LAYER ann_layerH[], LAYER *ann_layerO, WEIGHT_LINE ann_weight[], ARCHITECTURE network )     //給權重、門檻值亂數
{
    int i;

    // 給予隱藏層 theta 亂數
        for( i=0; i<network.layer-2; i++ )
        {
            Array2dRandom_double( &ann_layerH[i].theta , RAND_THETA_MAX, RAND_THETA_MIN );
            printf("ann_LayerH_theta = \n");
            Array2dPrintf_double( ann_layerH[i].theta );
        }

    // 給予輸出層 theta 亂數
        Array2dRandom_double( &ann_layerO->theta, RAND_THETA_MAX, RAND_THETA_MIN );
        printf("ann_LayerO_theta = \n");
        Array2dPrintf_double( ann_layerO->theta );

    // 給予 weight 亂數
        for( i=0; i<network.layer-1; i++ )
        {
            Array2dRandom_double( &ann_weight[i].weight, RAND_W_MAX, RAND_W_MIN );
            printf("ann_weight = \n");
            Array2dPrintf_double( ann_weight[i].weight );
        }

    return 1;
}

int SaveAnn( char *filename, LAYER ann_layerH[], LAYER ann_layerO, WEIGHT_LINE ann_weight[], ARCHITECTURE network  )    //儲存權重、門檻值
{
    int i, j, layer;
    FILE* fp_save = fopen(filename,"w");

    if( fp_save == NULL )
        return 0;

    /*--- 儲存結構 ---*/
        fprintf( fp_save,"Input_Layer:%d\n", network.nodeI );   //輸入層
        fprintf( fp_save,"Hidden_Layer:" );                     //隱藏層
        for( i=0; i<network.layer-2; i++ )
        {
            fprintf(fp_save,"%d",network.nodeH[i]);
            if( i != network.layer-3 )
                fprintf(fp_save,",");
        }
        fprintf( fp_save,"\nOutput_Layer:%d\n", network.nodeO );//輸出層

    /*--- 儲存 theta ---*/
        //儲存隱藏層 theta
            fprintf( fp_save, "ann_LayerH_threshold\n");
            for( layer=0; layer<network.layer-2; layer++ ){
                for( i=0; i<ann_layerH[layer].theta.row; i++ ){
                    for( j=0; j<ann_layerH[layer].theta.col; j++ ){
                        fprintf( fp_save, "%.2lf,", ann_layerH[layer].theta.arr[i][j] );
                    }
                    fprintf( fp_save, "\n" );
                }
            }

        //儲存輸出層 theta
            fprintf( fp_save, "ann_LayerO_threshold\n");
            for( i=0; i<ann_layerO.theta.row; i++ ){
                for( j=0; j<ann_layerO.theta.col; j++ ){
                    fprintf( fp_save, "%.2lf,", ann_layerO.theta.arr[i][j] );
                }
                fprintf( fp_save, "\n" );
            }

    /*--- 儲存 weight ---*/
        fprintf( fp_save, "ann_weight\n");
        for( layer=0; layer<network.layer-1; layer++ ){
            for( i=0; i<ann_weight[layer].weight.row; i++ ){
                for( j=0; j<ann_weight[layer].weight.col; j++ ){
                    fprintf( fp_save, "%.2lf,", ann_weight[layer].weight.arr[i][j] );
                }
                fprintf( fp_save, "\n" );
            }
        }

    fclose(fp_save);
    return 1;
}

int LoadAnn( char *filename, LAYER ann_layerH[], LAYER *ann_layerO, WEIGHT_LINE ann_weight[], ARCHITECTURE network  )   //讀取權重、門檻值
{
    int i, j, layer;
    FILE* fp_load ;
    ARCHITECTURE network_fromFile;
    int temp;
    char tempc[2];

    LoadNetworkArchitecture( filename, &network_fromFile );

    /*--- 檢查結構是否相同 ---*/
        if( !Architecture_Compare( network, network_fromFile ) ){
            printf("ERROR!! Architecture is different.");
            return 0;
        }
        else{   //結構相同則開始讀檔
            fp_load = fopen( filename, "r");
            if( fp_load == NULL ){
                printf("ERROR!! No file %s",filename);
                return 0;
            }
            fscanf( fp_load,"Input_Layer:%d\n", &temp );   //輸入層
            fscanf( fp_load,"Hidden_Layer:" );                     //隱藏層
            for( i=0; i<network.layer-2; i++ ){
                fscanf( fp_load,"%d", &temp );
                if( i != network.layer-3 )
                    fscanf( fp_load,",", tempc );
            }
            fscanf( fp_load,"\nOutput_Layer:%d\n", &temp );//輸出層
        }


    /*--- 讀取 theta ---*/
        //讀取隱藏層 theta
            fscanf( fp_load, "ann_LayerH_threshold\n");
            for( layer=0; layer<network.layer-2; layer++ ){
                for( i=0; i<ann_layerH[layer].theta.row; i++ ){
                    for( j=0; j<ann_layerH[layer].theta.col; j++ ){
                        fscanf( fp_load, "%lf,", &ann_layerH[layer].theta.arr[i][j] );
                    }
                    fscanf( fp_load, "\n" );
                }
            }

        //讀取輸出層 theta
            fscanf( fp_load, "ann_LayerO_threshold\n");
            for( i=0; i<ann_layerO->theta.row; i++ ){
                for( j=0; j<ann_layerO->theta.col; j++ ){
                    fscanf( fp_load, "%lf,", &ann_layerO->theta.arr[i][j] );
                }
                fscanf( fp_load, "\n" );
            }

    /*--- 儲存 weight ---*/
        fscanf( fp_load, "ann_weight\n");
        for( layer=0; layer<network.layer-1; layer++ ){
            for( i=0; i<ann_weight[layer].weight.row; i++ ){
                for( j=0; j<ann_weight[layer].weight.col; j++ ){
                    fscanf( fp_load, "%lf,", &ann_weight[layer].weight.arr[i][j] );
                }
                fscanf( fp_load, "\n" );
            }
        }

    fclose(fp_load);
    return 1;
}

int Architecture_Compare( ARCHITECTURE a, ARCHITECTURE b)   //比較兩個網路架構是否相同
{
    int l;

    if( a.layer != b.layer || a.nodeI != b.nodeI || a.nodeO != b.nodeO )
        return 0;
    for( l=0; l<a.layer; l++ ){
        if( a.nodeH[l] != b.nodeH[l] )
        return 0;
    }

    return 1;
}

int LoadTrainData( char *filename, TRAINING_DATA *ann_data, ARCHITECTURE network )  //讀取訓練資料
{
    FILE* fp_train = fopen(filename,"r");
    char check = 0;
    char buffer[256];
    int dataRow = 0;    //讀取的資料筆數計數器
    int dataCol = 0;    //讀取的資料維度計數器

    if( fp_train == NULL)
    {
        printf( "No file %s\n", filename );
        return 0;
    }

    /*--- 配置資料的動態陣列 ---*/
        //配置 feature 動態記憶體
            malloc2d_double( &ann_data->feature, ann_data->data_size, network.nodeI );
        //配置 target 動態記憶體
            malloc2d_double( &ann_data->target, ann_data->data_size, network.nodeO );

    /*--- 讀取資料 ---*/
    while( check != EOF )
    {
        check = 0;
        while( check != '\n' && check != EOF && dataRow != ann_data->data_size )
        {
            /*讀數字*/
                fscanf( fp_train, "%[0-9 .-]", buffer );        //只讀 0-9 空白 '.' '-'
                check = fgetc( fp_train );          //檢查是否要換行

            /*將資料放到陣列中*/
                if( dataCol < network.nodeI )       // feature 讀取，每一個逗點當作一個維度讀取
                    ann_data->feature.arr[dataRow][dataCol] = atof( buffer );
                else if( dataCol < network.nodeI + network.nodeO )       // feature 讀取完畢，開始 target 讀取，每一個逗點當作一個維度讀取
                    ann_data->target.arr[dataRow][dataCol-network.nodeI] = atof( buffer );
                else{                                                               //資料維度超過輸入、輸出層節點總和
                    printf("Error!!%s Overload at %d row! %d col\n",filename,dataRow+1,dataCol);
                    return 0;}

            dataCol++;
        }

        if( dataCol < network.nodeI + network.nodeO )       //資料維度小於輸入、輸出層節點總和
        {
            printf("Error!!%s data is less %d at %d row!\n", filename, network.nodeI + network.nodeO, dataRow+1 );
            return 0;
        }

        dataRow++;      //換下一行
        dataCol=0;      //重新從第一個維度抓資料
    }

    return 1;
}
