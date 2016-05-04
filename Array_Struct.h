#ifndef H_ARRAY_STRUCT_H
#define H_ARRAY_STRUCT_H


typedef struct ARRAY_2D_
{
    int row;
    int col;
    double** arr;
}ARRAY_2D;

typedef struct LAYER_
{
	int node;
	ARRAY_2D input;
	ARRAY_2D output;
	ARRAY_2D theta;
	ARRAY_2D theta_delta;
	int activationNum;
}LAYER;

typedef struct WEIGHT_LINE_
{
	ARRAY_2D weight;
	ARRAY_2D weight_delta;
}WEIGHT_LINE;

#endif
