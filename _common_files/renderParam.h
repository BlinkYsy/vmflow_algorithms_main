#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include "commStu.h"

using std::string;
using std::vector;

//渲染参数
class renderParam
{
public:
	vector<cv::RotatedRect>		m_renderRectBox;						//渲染矩形
	vector<circleROI>			m_renderCircle;							//渲染圆形
	vector<vector<cv::Point2f>>	m_renderPolygon;						//渲染多边形
	vector<cv::Point2f>			m_renderDot;							//渲染点集
	vector<linePoints>			m_renderLine;							//渲染直线
	vector<cv::Point2f>			m_renderDelDot;							//渲染剔除点集

public:
	string conRenderRect();
	string conRenderCircle();
	string conRenderPolygon();
	string conRenderDot();
	string conRenderLine();
	string conRenderDelDot();
	void conOutImg(cv::Mat inImg, void*& outImg);
};
