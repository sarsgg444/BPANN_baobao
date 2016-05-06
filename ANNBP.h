#ifndef H_ANNBP
#define H_ANNBP

#include "Array_Struct.h"

///亂數上下限
#define RAND_W_MAX 1.0     //亂數權重的最大值
#define RAND_W_MIN -1.0    //亂數權重的最小值
#define RAND_THETA_MAX 10   //亂數門檻的最大值
#define RAND_THETA_MIN -10  //亂數門檻的最小值

///ActivationFunction
//上下限
#define ACTIVATIONFUNCTION_UPPER_LIMIT ACT_MAX
#define ACTIVATIONFUNCTION_LOWER_LIMIT ACT_MIN
#define ACT_MAX 1
#define ACT_MIN 0
//種類
enum{NON,SIGMOID,STEP};



char LoadTrainData( FILE* trainData, ARRAY_2D *ann_LayerI_input, ARRAY_2D *ann_LayerT_output );

//激活函數
double ActivationFunction( double x, int function_switch );

int ANNBP_LayerA_ErrorValue											/*--- ANNBP_LayerA_ErrorValue 說明 ---*/
(   ARRAY_2D *ann_LayerA_output,			// ann_LayerA_output A層的輸出值
    ARRAY_2D *ann_LayerAB_weight,			// ann_LayerAB_weight AB層之間的權重
    ARRAY_2D *ann_LayerB_error,		    	// ann_LayerB_error B層的誤差項(已知，A的右邊(後面)一層)
    ARRAY_2D *ann_LayerA_error	        	// ann_LayerA_error A層的誤差項(未知，要計算求得)
);

int ANNBP_FixWeight
(   ARRAY_2D *ann_LayerA_output,
    ARRAY_2D *ann_LayerB_error,
    ARRAY_2D *ann_LayerAB_weightDelta,
    double learnSpeed
);

//儲存網路門檻值、權重值
int SaveAnn
(   char *filename,
    LAYER ann_layerH[], LAYER ann_layerO,
    WEIGHT_LINE ann_weight[],
    ARCHITECTURE network
);

//讀取網路門檻值、權重值
int LoadAnn
(   char *filename,
    LAYER ann_layerH[],
    LAYER *ann_layerO,
    WEIGHT_LINE ann_weight[],
    ARCHITECTURE network
 );

//讀取網路架構:層數及各層節點數
int LoadNetworkArchitecture( char *filename, ARCHITECTURE *network );

//讀取訓練資料
int LoadTrainData( char *filename ,TRAINING_DATA *ann_data, ARCHITECTURE network );

//每一層節點的記憶體配置
int Init_ann_layerI( LAYER* ann_layerI, ARCHITECTURE network );
int Init_ann_layerO( LAYER* ann_layerO, ARCHITECTURE network );
int Init_ann_layerH( LAYER** ann_layerH, ARCHITECTURE network );

//每一層權重的記憶體配置
int Init_ann_weight( WEIGHT_LINE** weight, ARCHITECTURE network );

//亂數給予門檻、權重
int Random_ANN( LAYER ann_layerH[], LAYER *ann_layerO, WEIGHT_LINE ann_weight[], ARCHITECTURE network );

//比較兩個網路結構是否相同
int Architecture_Compare( ARCHITECTURE a, ARCHITECTURE b);

//從資料庫讀取資料到輸入層
int PutRowTrainData( LAYER* ann_layerI, TRAINING_DATA ann_data, int loadRow );

//計算 ann_layerA 的輸出經過 WEIGHT_LINE 計算的結果
int ANNBP_LineCompute( LAYER ann_layerA, WEIGHT_LINE weight ,LAYER *ann_layerB );

//計算每一層節點的 output (input->theta->activation->output)
int ANNBP_LayerCompute( LAYER *ann_layerA );

#endif // H_ANNBP
