#pragma once

#include <iostream>
#include "../_common_files/commStu.h"
#include "../_common_files/parseBaseParam.h"
#include "../_common_files/renderParam.h"

#define PI       3.14159265358979323846

using namespace std;

class CTraColor :parseBaseParam
{
public:
	int m_colorTransformType = 0;												//转换类型
	int m_showChannel = 0;														//显示通道
	cv::Mat m_iniImg;
	cv::Mat m_traColorImg;
	int m_moduStatus = 0;
	renderParam m_renderparam;

public:
	int	parseParam(void* inImg, char* xmlIn);
	int cpr_transformColor(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
	int transformColor(void*& outImg);
	int traColor(cv::Mat inImg, cv::Mat& outImg);
private:
	int	traColorModuleResult(cv::Mat inImg, void*& outImg);
	int	outputTransformColorXml(char* xmlOut, string funName);
	int traRGB2YUV(cv::Mat inImg, cv::Mat& outImg);
	int traRGB2HSI(cv::Mat inImg, cv::Mat& outImg);
	int traRGB2HSV(cv::Mat inImg, cv::Mat& outImg);
	int traRGB2GRAY(cv::Mat inImg, cv::Mat& outImg);

};