#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include "commStu.h"

using std::string;
using std::vector;

//基本参数基类
class parseBaseParam
{
public:
	cv::Mat						m_inImg;								//输入图像
	int							m_debugProcess = 0;						//过程调试
	int							m_debugResult = 0;						//结果调试
	vector<cv::RotatedRect>		m_vecRectROI;							//矩形区域
	vector<circleROI>			m_vecCircleROI;							//圆形区域
	vector<vector<cv::Point>>	m_vecPolygonROI;						//多边形区域
	vector<vector<cv::Point>>	m_vecDelConROI;							//屏蔽区域
	float						m_polygonAngle = 0;						//多边形角度

public:
	int	resBaseParam(void* inImage, string name, char* xmlIn);
	int	iniBaseParamImg(cv::Mat inImage, cv::Mat& outImage);
	cv::Rect pointToRect(const vector<cv::Point> inContours);
	vector<cv::Point> rotatePolygon(vector<cv::Point> inContours, cv::Rect inRect, float rotateAngle);
};
