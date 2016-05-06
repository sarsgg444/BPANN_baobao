#ifndef H_ARRAY_STRUCT_H
#define H_ARRAY_STRUCT_H

#define MAX_HIDDEN_LAYER 1000

typedef struct ARRAY_2D_
{
    int row;
    int col;
    double** arr;
}ARRAY_2D;

typedef struct ARCHITECTURE_
{
    int layer;
    int nodeI;
    int *nodeH;
    int nodeO;
}ARCHITECTURE;

typedef struct LAYER_
{
	int node;
	ARRAY_2D input;
	ARRAY_2D output;
	ARRAY_2D theta;
	ARRAY_2D theta_delta;
	ARRAY_2D error;
	int activationNum;
}LAYER;

typedef struct WEIGHT_LINE_
{
	ARRAY_2D weight;
	ARRAY_2D weight_delta;
}WEIGHT_LINE;

typedef struct TRAINING_DATA_
{
    ARRAY_2D feature;      //存放每一筆訓練特徵的輸入值，以2維陣列的方式存放。feature[n][m]，第 n 筆資料第 m 個維度.
    ARRAY_2D target;       //存放每一筆訓練特徵的輸出值，以2維陣列的方式存放。target[n][m]，第 n 筆資料第 m 個維度.
    double data_size;       //總資料筆數
}TRAINING_DATA;

#endif
