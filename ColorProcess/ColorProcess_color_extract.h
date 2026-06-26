#pragma once

#include <iostream>
#include "../_common_files/commStu.h"
#include "../_common_files/parseBaseParam.h"
#include "../_common_files/renderParam.h"

using namespace std;

//运行参数
struct extColorRunParam
{
	int							colorSpace = 0;							//颜色空间
	bool						colorReversal = 0;						//颜色反转
	int							channelLow_1 = 0;						//通道1低阈值
	int							channelHigh_1 = 255;					//通道1高阈值
	int							channelLow_2 = 0;						//通道2低阈值
	int							channelHigh_2 = 255;					//通道2高阈值
	int							channelLow_3 = 0;						//通道3低阈值
	int							channelHigh_3 = 255;					//通道3高阈值
};

//结果显示
struct extColorResultParam
{

};

//模块结果
struct extColorModuleResult
{
	bool						moduStatus = false;						//模块状态
	int							colorTotalArea = 0;						//颜色总面积
};

class CExtColor :parseBaseParam
{
public:
	extColorRunParam m_runparam;
	extColorResultParam m_resultparam;
	extColorModuleResult m_moduleresult;
	renderParam m_renderparam;
	cv::Mat m_iniImg;
	cv::Mat m_extColorImg;
public:
	int cpr_extractColor(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
	int	parseRunParam(void* inImg, char* xmlIn);
	int	parseResultParam(void* inImg, char* xmlIn);
	int	parseParam(void* inImg, char* xmlIn);
private:
	int	extractColor(void*& outImg);
	int extColor(cv::Mat inImg, cv::Mat& outImg);
	int	extColorModuleResult(cv::Mat inImg, void*& outImg);
	int areaContours(cv::Mat inImg);
	int	outputExtractColorXml(char* xmlOut, string funName);
	int rgbToHSI(cv::Mat inImg, cv::Mat& outImg);
};