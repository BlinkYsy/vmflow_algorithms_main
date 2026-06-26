#pragma once
#include <iostream>
#include <string>
#include "Markup.h"
#include <opencv2/opencv.hpp>
#include "commStu.h"
//#include "CFileLog.h"

using std::string;
using std::vector;
using namespace cv;

class ResolveXml
{
public:
	string float2string(float Num);
	string cvRect2string(cv::Rect rect);
	string cvPts2string(vector<cv::Point>& vePt);

public:
	int parseXmlDouble(string strBody, string param, string item, string childItem, vector<double>& veDouble);
	double parseXmlDou(std::string strBody, string param, string item, string childItem);
	int parseXmlInt(string strBody, string param, string item, string childItem);
	int parseXmlVePts(string strBody, string param, string item, string childitem, vector<cv::Point>& vePts);
	int parseXmlVePts2f(string strBody, string param, string item, string childitem, std::vector<cv::Point2f>& vePts);
	int parseXmlPt(string strBody, string param, string item, string childitem, cv::Point& pt);
	int parseXmlPt2d(string strBody, string param, string item, string childitem, cv::Point2d& pt);
	int parseXmlRect(string strBody, string param, string item, string childitem, cv::Rect& rect);
	int parseXmlRectROI(string strBody, string param, string item, string childitem, vector<cv::RotatedRect>& vecRectROI);
	int parseXmlVecRectROI(string strBody, string param, string item, string childitem, vector<cv::Rect>& vecRectROI);
	int parseXmlCircleROI(string strBody, string param, string item, string childitem, vector<circleROI>& vecCircleROI);
	int parseXmlPolygonROI(string strBody, string param, string item, string childitem, vector<vector<cv::Point>>& vecPolygonROI);
	
	int parseXmlMaskCropAreas(string strBody, string param, string item, string childitem, cv::Rect& rect, int& selectShape, int& selectColor);
	string parseXmlStr(std::string strBody, string param, string item, string childItem);
	string parseXmlInterfaceName(std::string strBody, string param);

	// xml½á¹ûÊä³ö
	int saveLocalCircleXml(string& xmlOut, string funName,cv::Point2f center, double radius,std::vector<cv::Point> contourPts);
	int saveLocalMatchXml(string& xmlOut, string funName, vector<string> matchResult, cv::Point pt);
};

