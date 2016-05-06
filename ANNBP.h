#ifndef H_ANNBP
#define H_ANNBP

#include "Array_Struct.h"

///�üƤW�U��
#define RAND_W_MAX 1.0     //�ü��v�����̤j��
#define RAND_W_MIN -1.0    //�ü��v�����̤p��
#define RAND_THETA_MAX 10   //�üƪ��e���̤j��
#define RAND_THETA_MIN -10  //�üƪ��e���̤p��

///ActivationFunction
//�W�U��
#define ACTIVATIONFUNCTION_UPPER_LIMIT ACT_MAX
#define ACTIVATIONFUNCTION_LOWER_LIMIT ACT_MIN
#define ACT_MAX 1
#define ACT_MIN 0
//����
enum{NON,SIGMOID,STEP};



char LoadTrainData( FILE* trainData, ARRAY_2D *ann_LayerI_input, ARRAY_2D *ann_LayerT_output );

//�E�����
double ActivationFunction( double x, int function_switch );

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

//�x�s�������e�ȡB�v����
int SaveAnn
(   char *filename,
    LAYER ann_layerH[], LAYER ann_layerO,
    WEIGHT_LINE ann_weight[],
    ARCHITECTURE network
);

//Ū���������e�ȡB�v����
int LoadAnn
(   char *filename,
    LAYER ann_layerH[],
    LAYER *ann_layerO,
    WEIGHT_LINE ann_weight[],
    ARCHITECTURE network
 );

//Ū�������[�c:�h�ƤΦU�h�`�I��
int LoadNetworkArchitecture( char *filename, ARCHITECTURE *network );

//Ū���V�m���
int LoadTrainData( char *filename ,TRAINING_DATA *ann_data, ARCHITECTURE network );

//�C�@�h�`�I���O����t�m
int Init_ann_layerI( LAYER* ann_layerI, ARCHITECTURE network );
int Init_ann_layerO( LAYER* ann_layerO, ARCHITECTURE network );
int Init_ann_layerH( LAYER** ann_layerH, ARCHITECTURE network );

//�C�@�h�v�����O����t�m
int Init_ann_weight( WEIGHT_LINE** weight, ARCHITECTURE network );

//�üƵ������e�B�v��
int Random_ANN( LAYER ann_layerH[], LAYER *ann_layerO, WEIGHT_LINE ann_weight[], ARCHITECTURE network );

//�����Ӻ������c�O�_�ۦP
int Architecture_Compare( ARCHITECTURE a, ARCHITECTURE b);

//�q��ƮwŪ����ƨ��J�h
int PutRowTrainData( LAYER* ann_layerI, TRAINING_DATA ann_data, int loadRow );

//�p�� ann_layerA ����X�g�L WEIGHT_LINE �p�⪺���G
int ANNBP_LineCompute( LAYER ann_layerA, WEIGHT_LINE weight ,LAYER *ann_layerB );

//�p��C�@�h�`�I�� output (input->theta->activation->output)
int ANNBP_LayerCompute( LAYER *ann_layerA );

#endif // H_ANNBP
