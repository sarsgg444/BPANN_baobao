#ifndef H_ANNBP
#define H_ANNBP

#include "Array_Struct.h"

char LoadTrainData( FILE* trainData, ARRAY_2D *ann_LayerI_input, ARRAY_2D *ann_LayerT_output );
int ActivationFunction( ARRAY_2D *arrayA, ARRAY_2D *arrayOut );

int ANNBP_LayerA_ErrorValue											/*--- ANNBP_LayerA_ErrorValue ���� ---*/
(   ARRAY_2D *ann_LayerA_output,			// ann_LayerA_output A�h����X��
    ARRAY_2D *ann_LayerAB_weight,			// ann_LayerAB_weight AB�h�������v��
    ARRAY_2D *ann_LayerB_error,		    	// ann_LayerB_error B�h���~�t��(�w���AA���k��(�᭱)�@�h)
    ARRAY_2D *ann_LayerA_error	        	// ann_LayerA_error A�h���~�t��(�����A�n�p��D�o)
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
