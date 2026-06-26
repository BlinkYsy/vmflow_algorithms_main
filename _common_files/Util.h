#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define INF  0xFFFFFFFF

using std::string;
using std::vector;
//using namespace cv;

class Util
{
public:
	void SplitString(const string& s, vector<string>& v, const string& c);

	string getRngStr();

	// 图像 二值化处理基础算子
	int otsu(cv::Mat srcgray);
	double disPP(cv::Point pt1, cv::Point pt2);
	double disPPDouble(double p1, double p2, double p3, double p4);
	int pts2Rect(cv::Rect& rect, cv::Point pt1, cv::Point pt2, cv::Point pt3, cv::Point pt4);

	float calAreaFan(float angle, float raduis);	//	计算扇形面积
	int   mat2CharS(cv::Mat img, char* outImg);

	int pp2Line(cv::Vec4f& line,cv::Point pt1, cv::Point pt2);		// 两点直线方程
	double ptLineDis(cv::Point2d point, cv::Point2d pnt1, cv::Point2d pnt2);
	void ptStartEnd(cv::Point2d& pt1, cv::Point2d& pt2);
	int P2LCross(const cv::Point2f linePt1, const cv::Point2f linePt2, const cv::Point2f point, cv::Point2f& crossPt);
	double point2LineDistance(cv::Point point, cv::Vec4f lineParam);//计算点到直线的距离
	cv::Point2d get2lineIPoint(cv::Vec4f lineParam1, cv::Vec4f lineParam2);//两条直线的交点

	int warpPt(cv::Point pt1, cv::Point center, double alpha, cv::Point& ptOut);
	int warpPt2d(cv::Point2d pt1, cv::Point2d center, double alpha, cv::Point2d& ptOut);	// 坐标旋转
	void warpImgMat(cv::Mat& src, cv::Mat& dst, double angle,int dafult = 0);		// angle 为角度值

	int char2Mat(cv::Mat& outImg, char* imgdata, int width, int height, int channel);

	int pt3Area(cv::Point pt1, cv::Point2f pt2, cv::Point2f pt3);
	bool isPointInRect(cv::Point pt, cv::Point lbPt, cv::Point ltPt, cv::Point rtPt, cv::Point rbPt);
	bool isPointInRectEx(cv::Point pt, int stdS,cv::Point2f lbPt, cv::Point2f ltPt, cv::Point2f rtPt, cv::Point2f rbPt);	//

	void pt2CaliperPts(cv::Point pt, double angle, int caliperLengh, int caliperWidth, cv::Point& ltPt, cv::Point& rtPt, cv::Point& lbPt, cv::Point& rbPt);

	void cleanFOlderImgFile(char* imageFolder);
	double lineLineAngle(double p1x, double p1y, double p2x, double p2y, double p3x, double p3y, double p4x, double p4y, int flog);
};
