//#include "pch.h"
#include "Util.h"
#include <time.h>
#include <random>

double Util::lineLineAngle(double p1x, double p1y, double p2x, double p2y, double p3x, double p3y, double p4x, double p4y, int flog)
{
	//获取两条直线的向量坐标
	double v1x = p2x - p1x;
	double v1y = p2y - p1y;
	double v2x = p4x - p3x;
	double v2y = p4y - p3y;
	//初始化局部变量
	float theta = 0.0;
	double err = 0.00001;
	double line1, line2;

	if (flog == 1)//矢量直线，夹角取值范围[-π,π)
	{
		theta = atan2(v2y, v2x) - atan2(v1y, v1x);//atan2方便计算出角度准确处于哪一个象限
		//atan2的取值范围是[−π,π]，在进行相减之后得到的夹角是在[−2π,2π]
		if (theta > CV_PI)
			theta -= 2 * CV_PI;//当结果大于π时，对结果减去2π
		if (theta < -CV_PI)
			theta += 2 * CV_PI; //当结果小于−π时，对结果加上2π
		theta = theta * 180.0 / CV_PI;
		if (theta == 180)
			theta = -180;//当两条直线水平且方向相反，默认为-180度
		return theta;
	}
	if (flog == 0)//二维直线，夹角取值范围[0,π)
	{
		line1 = sqrt(v1x * v1x + v1y * v1y);
		line2 = sqrt(v2x * v2x + v2y * v2y);
		if ((line1 > err) && (line2 > err))
			theta = acos((v1x * v2x + v1y * v2y) / (line1 * line2));//θ=acos(v1·v2/||v1||||v2||)
		theta = theta * 180.0 / CV_PI;
		if (theta > 90)
			theta = 180 - theta;//当两条直线水平，默认为0度
		return theta;
	}
}

void Util::SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

int Util::otsu(cv::Mat srcgray)
{
	const int Grayscale = 256;
	int graynum[Grayscale] = { 0 };
	int r = srcgray.rows;
	int c = srcgray.cols;
	for (int i = 0; i < r; ++i)
	{
		const uchar* ptr = srcgray.ptr<uchar>(i);
		for (int j = 0; j < c; ++j) {        //直方图统计
			graynum[ptr[j]]++;
		}
	}

	double P[Grayscale] = { 0 };
	double PK[Grayscale] = { 0 };
	double MK[Grayscale] = { 0 };
	double srcpixnum = r * c, sumtmpPK = 0, sumtmpMK = 0;
	for (int i = 0; i < Grayscale; ++i)
	{
		P[i] = graynum[i] / srcpixnum;   //每个灰度级出现的概率
		PK[i] = sumtmpPK + P[i];         //概率累计和 
		sumtmpPK = PK[i];
		MK[i] = sumtmpMK + i * P[i];       //灰度级的累加均值                                                                                                                                                                                                                                                                                                                                                                                                        
		sumtmpMK = MK[i];
	}

	//计算类间方差
	double Var = 0;
	int thresh = 0;
	for (int k = 0; k < Grayscale; ++k)
	{
		if ((MK[Grayscale - 1] * PK[k] - MK[k]) * (MK[Grayscale - 1] * PK[k] - MK[k]) / (PK[k] * (1 - PK[k])) > Var) {
			Var = (MK[Grayscale - 1] * PK[k] - MK[k]) * (MK[Grayscale - 1] * PK[k] - MK[k]) / (PK[k] * (1 - PK[k]));
			thresh = k;
		}
	}

	return thresh;
}

double Util::disPP(cv::Point pt1, cv::Point pt2)
{
	double ret = 0;
	ret = sqrt((double)(pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));
	return ret;
}

double Util::disPPDouble(double p1, double p2, double p3, double p4)
{
	double ret = 0;
	ret = sqrt((double)(p1 - p3) * (p1 - p3) + (p2 - p4) * (p2 - p4));
	return ret;
}

int Util::P2LCross(const cv::Point2f linePt1, const cv::Point2f linePt2, const cv::Point2f point, cv::Point2f& crossPt)
{
	//float k = (linePt1.y - linePt2.y) / (linePt1.x - linePt2.x);
	//float b = (linePt1.y - k * linePt1.y);

	//float m = point.x + k * point.y;

	///// 求两直线交点坐标
	//crossPt.x = ((m - A * B) / (A * A + 1));
	//crossPt.y = (A * crossPt.x + B);

	float k = 0.0;
	if (linePt1.x == linePt2.x)
	{
		crossPt.x = linePt1.x;
		crossPt.y = point.y;
	}
	else
	{
		k = (linePt1.y - linePt2.y) * 1.0 / (linePt1.x - linePt2.x);
		float A = k;
		float B = -1.0;
		float C = linePt1.y - k * linePt1.x;

		crossPt.x = (B * B * point.x - A * B * point.y - A * C) / (A * A + B * B);
		crossPt.y = (A * A * point.y - A * B * point.x - B * C) / (A * A + B * B);
	}
	return 0;
}

int Util::pts2Rect(cv::Rect& rect, cv::Point pt1, cv::Point pt2, cv::Point pt3, cv::Point pt4)
{
	int ret = -1;

	int topLeft_x = cv::min(cv::min((int)(cv::min(pt1.x, pt2.x)), pt3.x), pt4.x);
	int topLeft_y = cv::min(cv::min((int)(cv::min(pt1.y, pt2.y)), pt3.y), pt4.y);

	int butRight_x = cv::max(cv::max((int)(cv::max(pt1.x, pt2.x)), pt3.x), pt4.x);
	int butRight_y = cv::max(cv::max((int)(cv::max(pt1.y, pt2.y)), pt3.y), pt4.y);

	rect = cv::Rect(topLeft_x, topLeft_y, (butRight_x - topLeft_x), (butRight_y - topLeft_y));
	return 0;
}

float Util::calAreaFan(float angle, float raduis)
{
	float S = float(CV_PI * raduis * raduis * angle / 360.0);
	return S;
}

int   Util::mat2CharS(cv::Mat img, char* outImg)
{
	std::vector<uchar> buf;
	cv::imencode(".bmp", img, buf); // 将Mat以BMP格式存入uchar的buf数组中

	// 使用 std::copy 算法，一行代码完成拷贝
	std::copy(buf.begin(), buf.end(), outImg);

	return 0;

	/*
	std::vector<uchar> buf;
	cv::imencode(".bmp", img, buf);		//将Mat以BMP格式存入uchar的buf数组中
	for each (uchar var in buf)				//将buf拷贝到C#的byte[]中
	{
		*outImg = var;
		outImg++;
	}
	return 0;
	*/
}

int Util::pp2Line(cv::Vec4f& line, cv::Point pt1, cv::Point pt2)		// 点斜式
{
	float k = 0.0;
	float b = 0.0;
	if (pt1.x == pt2.x)
	{
		k = INF;
		b = (float)pt1.x;
	}
	else
	{
		k = (float)(pt1.y - pt2.y) / (float)(pt1.x - pt2.x);
		b = pt1.y - k * pt1.x;
	}
	line = cv::Vec4f(k, b, (float)pt1.x, (float)pt1.y);
	return 0;
}

double Util::ptLineDis(cv::Point2d point, cv::Point2d pnt1, cv::Point2d pnt2)
{
	double ptLineDis = 0.0;
	ptStartEnd(pnt1, pnt2);

	double A = (double)pnt1.y - (double)pnt2.y;
	double B = (double)pnt2.x - (double)pnt1.x;
	double C = (double)pnt1.y * (-B) + (double)pnt1.x * (-A);

	double D = sqrt(A * A + B * B);
	double E = abs(A * point.x + B * point.y + C);

	if (D == 0)
		ptLineDis = 10000;	// 图像长宽不超过10000个像素点
	else
		ptLineDis = (float)E / D;

	return ptLineDis;
}
void Util::ptStartEnd(cv::Point2d& pt1, cv::Point2d& pt2)
{
	cv::Point2d ptS, ptE;
	if (pt1.x < pt2.x)
	{
		ptS = pt1;
		ptE = pt2;
	}
	else if (pt1.x > pt2.x)
	{
		ptS = pt2;
		ptE = pt1;
	}
	else if (pt1.x == pt2.x)
	{
		if (pt1.y < pt2.y)
		{
			ptS = pt1;
			ptE = pt2;
		}
		else
		{
			ptS = pt2;
			ptE = pt1;
		}
	}
	pt1 = ptS;
	pt2 = ptE;
}


int Util::warpPt2d(cv::Point2d pt1, cv::Point2d center, double alpha, cv::Point2d& ptOut)
{
	// 逆时针旋转
	double x = (pt1.x - center.x) * cos(alpha) - (pt1.y - center.y) * sin(alpha) + center.x;
	double y = (pt1.x - center.x) * sin(alpha) + (pt1.y - center.y) * cos(alpha) + center.y;

	ptOut.x = x;
	ptOut.y = y;

	return 0;
}

int Util::warpPt(cv::Point pt1, cv::Point center, double alpha, cv::Point& ptOut)
{
	// 逆时针旋转
	double x = (pt1.x - center.x) * cos(alpha) - (pt1.y - center.y) * sin(alpha) + center.x;
	double y = (pt1.x - center.x) * sin(alpha) + (pt1.y - center.y) * cos(alpha) + center.y;

	ptOut.x = (int)x;
	ptOut.y = (int)y;

	return 0;
}

void Util::warpImgMat(cv::Mat& src, cv::Mat& dst, double angle, int dafult)
{
	/*if (angle == 360)
		angle = 0;*/

	double a = sin(angle * CV_PI / 180);
	double b = cos(angle * CV_PI / 180);
	int width = src.cols;
	int height = src.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	cv::Point center = cv::Point(src.cols / 2, src.rows / 2);
	cv::Mat map_matrix = getRotationMatrix2D(center, angle, 1.0);
	map_matrix.at<double>(0, 2) += (width_rotate - width) / 2;     // 修改坐标偏移
	map_matrix.at<double>(1, 2) += (height_rotate - height) / 2;   // 修改坐标偏移
	if (src.channels() == 3)
		warpAffine(src, dst, map_matrix, cv::Size(width_rotate, height_rotate), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(dafult, dafult, dafult));
	else if (src.channels() == 1)
		warpAffine(src, dst, map_matrix, cv::Size(width_rotate, height_rotate), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(dafult));

}

int Util::char2Mat(cv::Mat& outImg, char* imgdata, int width, int height, int channel)
{
	// 构造图像
	if (channel == 1)
	{
		outImg = cv::Mat(height, width, CV_MAKETYPE(CV_8UC2, channel), (void*)imgdata);
		cv::cvtColor(outImg, outImg, cv::COLOR_GRAY2BGR);
	}
	else if (channel == 3)
		outImg = cv::Mat(height, width, CV_MAKETYPE(CV_8UC3, channel), (void*)imgdata);
	else if (channel == 4)
	{
		outImg = cv::Mat(height, width, CV_MAKETYPE(CV_8UC4, channel), (void*)imgdata);
		cvtColor(outImg, outImg, cv::COLOR_RGBA2RGB);
	}
	else
		outImg = cv::Mat(height, width, CV_MAKETYPE(CV_8U, channel), (void*)imgdata);
	if (!outImg.data)
		return -2;
	return 0;
}

bool Util::isPointInRect(cv::Point pt, cv::Point lbPt, cv::Point ltPt, cv::Point rtPt, cv::Point rbPt)
{
	int x = pt.x;
	int y = pt.y;
	cv::Point A = lbPt;
	cv::Point B = ltPt;
	cv::Point C = rtPt;
	cv::Point D = rbPt;
	int a = (B.x - A.x) * (y - A.y) - (B.y - A.y) * (x - A.x);
	int b = (C.x - B.x) * (y - B.y) - (C.y - B.y) * (x - B.x);
	int c = (D.x - C.x) * (y - C.y) - (D.y - C.y) * (x - C.x);
	int d = (A.x - D.x) * (y - D.y) - (A.y - D.y) * (x - D.x);
	if ((a > 0 && b > 0 && c > 0 && d > 0) || (a < 0 && b < 0 && c < 0 && d < 0)) {
		return true;
	}

	//      AB X AP = (b.x - a.x, b.y - a.y) x (p.x - a.x, p.y - a.y) = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
	//      BC X BP = (c.x - b.x, c.y - b.y) x (p.x - b.x, p.y - b.y) = (c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x);
	return false;
}

// 三点坐标求面积
int Util::pt3Area(cv::Point pt1, cv::Point2f pt2, cv::Point2f pt3)
{
	int x1 = (int)pt1.x;
	int y1 = (int)pt1.y;

	int x2 = (int)pt2.x;
	int y2 = (int)pt2.y;

	int x3 = (int)pt3.x;
	int y3 = (int)pt3.y;

	int S = abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2;

	//int S = (pt1.x * pt2.y - pt1.x * pt3.y + pt2.x * pt3.y - pt2.x * pt1.y + pt2.x * pt1.y - pt2.x * pt2.y);
	return (int)S;
}

// 方案待优化
bool Util::isPointInRectEx(cv::Point pt, int stdS, cv::Point2f lbPt, cv::Point2f ltPt, cv::Point2f rtPt, cv::Point2f rbPt)
{
	int S1 = pt3Area(pt, lbPt, ltPt);
	int S2 = pt3Area(pt, ltPt, rtPt);
	int S3 = pt3Area(pt, rtPt, rbPt);
	int S4 = pt3Area(pt, rbPt, lbPt);

	int S = S1 + S2 + S3 + S4;
	//printf("%d %d %d %d	%d\n", S1, S2, S3, S4,S);
	if (S <= (stdS + 100))
		return true;
	else
		return false;

}


void Util::pt2CaliperPts(cv::Point pt, double angle, int caliperLengh, int caliperWidth, cv::Point& ltPt, cv::Point& rtPt, cv::Point& lbPt, cv::Point& rbPt)
{
	double sin_angle_b = angle * CV_PI / 180;

	cv::Point ptLeftTop, ptRightTop, ptLeftBuutom, ptRightButtom;
	ptLeftTop.x = pt.x - caliperWidth / 2;
	ptLeftTop.y = pt.y - caliperLengh / 2;

	ptRightTop.x = pt.x + caliperWidth / 2;
	ptRightTop.y = pt.y - caliperLengh / 2;

	ptLeftBuutom.x = pt.x - caliperWidth / 2;
	ptLeftBuutom.y = pt.y + caliperLengh / 2;

	ptRightButtom.x = pt.x + caliperWidth / 2;
	ptRightButtom.y = pt.y + caliperLengh / 2;

	// 旋转坐标
	warpPt(ptLeftTop, pt, sin_angle_b, ltPt);
	warpPt(ptRightTop, pt, sin_angle_b, rtPt);
	warpPt(ptLeftBuutom, pt, sin_angle_b, lbPt);
	warpPt(ptRightButtom, pt, sin_angle_b, rbPt);
}

string Util::getRngStr()
{
	/*SYSTEMTIME time;
	GetLocalTime(&time);
	char timeStr[512] = { 0 };
	sprintf(timeStr, "%d%d%d_%d%d%d_%d", time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond, time.wMilliseconds);
	return (string)timeStr;*/

	std::random_device rd;
	//for (int n = 0; n < 20000; ++n)
		//std::cout << rd() << std::endl;

	char timeStr[512] = { 0 };
	srand(rd());
	sprintf_s(timeStr, "%8d", rand());
	return (string)timeStr;
}

void Util::cleanFOlderImgFile(char* imageFolder)
{
	/*char cmd[128] = {0};
	sprintf(cmd, "start /b rd  %s /t", imageFolder);
	system(cmd);*/
}

double Util::point2LineDistance(cv::Point point, cv::Vec4f lineParam)
{
	cv::Point2d qIpoint;
	cv::Vec4f lineVert;
	lineVert[0] = -lineParam[0];
	//过点做直线的垂线，求垂线的 b
	lineVert[1] = point.y - lineVert[0] * point.x;
	lineVert[2] = point.x;
	lineVert[3] = point.y;
	//qIpoint = get2lineIPoint(lineParam, lineVert);
	//两线交点
	qIpoint.x = (lineParam[1] - lineVert[1]) / (lineVert[0] - lineParam[0]);
	qIpoint.y = lineVert[0] * qIpoint.x + lineVert[1];
	double dx = point.x - qIpoint.x;
	double dy = point.y - qIpoint.y;
	double fdis = sqrt(dx * dx + dy * dy);
	return fdis;
}

cv::Point2d Util::get2lineIPoint(cv::Vec4f lineParam1, cv::Vec4f lineParam2)
{
	//Vec4f :参数的前半部分给出的是直线的方向，而后半部分给出的是直线上的一点
	cv::Point2d result(-1, -1);

	double cos_theta = lineParam1[0];
	double sin_theta = lineParam1[1];
	double x = lineParam1[2];
	double y = lineParam1[3];
	double k = sin_theta / cos_theta;
	double b = y - k * x;

	cos_theta = lineParam2[0];
	sin_theta = lineParam2[1];
	x = lineParam2[2];
	y = lineParam2[3];
	double k1 = sin_theta / cos_theta;
	double b1 = y - k1 * x;

	result.x = (b1 - b) / (k - k1);
	result.y = k * x + b;

	return result;
}
