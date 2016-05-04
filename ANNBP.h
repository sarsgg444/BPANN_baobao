#ifndef H_ANNBP
#define H_ANNBP

#include "Array_Struct.h"

char LoadTrainData( FILE* trainData, ARRAY_2D *ann_LayerI_input, ARRAY_2D *ann_LayerT_output );
int ActivationFunction( ARRAY_2D *arrayA, ARRAY_2D *arrayOut );

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

int SaveAnn
(    FILE* SaveData,
     ARRAY_2D *ann_LayerIH_weight,
     ARRAY_2D *ann_LayerH_threshold,
     ARRAY_2D *ann_LayerHO_weight,
     ARRAY_2D *ann_LayerO_threshold
);

int LoadAnn
(    FILE* LoadData,
     ARRAY_2D *ann_LayerIH_weight,
     ARRAY_2D *ann_LayerH_threshold,
     ARRAY_2D *ann_LayerHO_weight,
     ARRAY_2D *ann_LayerO_threshold
);


#endif // H_ANNBP
