#pragma once
#include <cmath>
#include"basicStruct.h"

/*
*弱分类器结构体
*/
typedef struct MyStumpClassifier
{
	int compidx;

	float lerror; /* impurity of the right node */
	float rerror; /* impurity of the left  node */
	float error; /* 总的错误率*/
	float threshold;
	float left;
	float right;
} MyStumpClassifier;

typedef struct MyCARTClassifier
{
	/* number of internal nodes */
	int count;
	/* internal nodes (each array of <count> elements) */
	vector<MyStumpClassifier> classifier;
	float threshold;
} MyCARTClassifier;

typedef struct MyCascadeClassifier
{
	/* size of classifier */
	MySize size;
	/* internal nodes (each array of <count> elements) */
	vector<MyCARTClassifier> StrongClassifier;
} MyCascadeClassifier;