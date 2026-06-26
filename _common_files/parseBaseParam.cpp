//#include "pch.h"
#include "parseBaseParam.h"
#include "ResolveXml.h"
#include "errCode.h"

int	parseBaseParam::iniBaseParamImg(cv::Mat inImage, cv::Mat& outImage)
{
    int ret = 0;
    try
    {
        //指针创建图像更高效
        cv::Mat imgROI(inImage.rows, inImage.cols, inImage.type());
        unsigned char* data = imgROI.ptr<unsigned char>(0);
        std::memset(data, 0, inImage.rows * inImage.cols * sizeof(unsigned char));

        //矩形搜索区域
        for (size_t i = 0; i < m_vecRectROI.size(); i++)
        {
            RotatedRect rRect(Point2f(m_vecRectROI[i].center.x, m_vecRectROI[i].center.y), \
                Size2f(m_vecRectROI[i].size.width, m_vecRectROI[i].size.height), m_vecRectROI[i].angle);
            Point2f vecPts[4];
            rRect.points(vecPts);
            vector<vector<Point>> contours;
            vector<cv::Point> contour;
            contour.push_back(vecPts[0]);
            contour.push_back(vecPts[1]);
            contour.push_back(vecPts[2]);
            contour.push_back(vecPts[3]);
            contours.push_back(contour);
            cv::drawContours(imgROI, contours, 0, Scalar::all(255), -1);
            inImage.copyTo(outImage, imgROI);
        }

        //圆形搜索区域
        for (size_t i = 0; i < m_vecCircleROI.size(); i++)
        {
            circle(imgROI, cv::Point(m_vecCircleROI[i].centerX, m_vecCircleROI[i].centerY), \
                m_vecCircleROI[i].outerRadius, Scalar(255, 255, 255), -1);
            circle(imgROI, cv::Point(m_vecCircleROI[i].centerX, m_vecCircleROI[i].centerY), \
                m_vecCircleROI[i].innerRadius, Scalar(0, 0, 0), -1);
            inImage.copyTo(outImage, imgROI);
        }

        //多边形搜索区域
        for (size_t i = 0; i < m_vecPolygonROI.size(); i++)
        {
            vector<vector<cv::Point>> contours;
            cv::Rect rect = pointToRect(m_vecPolygonROI[i]);
            contours.push_back(rotatePolygon(m_vecPolygonROI[i], rect, m_polygonAngle));
            cv::drawContours(imgROI, contours, 0, Scalar::all(255), -1);
            inImage.copyTo(outImage, imgROI);
        }
        //无搜索区域时默认搜索区域为整张原图
        if (m_vecRectROI.size() == 0 && m_vecCircleROI.size() == 0 && m_vecPolygonROI.size() == 0)
            inImage.copyTo(outImage);

        //多边形屏蔽区域
        for (size_t i = 0; i < m_vecDelConROI.size(); i++)
        {
            vector<vector<cv::Point>> contours;
            cv::Rect rect = pointToRect(m_vecDelConROI[i]);
            contours.push_back(rotatePolygon(m_vecDelConROI[i], rect, m_polygonAngle));
            cv::drawContours(outImage, contours, 0, Scalar::all(0), -1);
        }
        imgROI.release();
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_iniBaseParamImg;
    }
    return ret;
}

int	parseBaseParam::resBaseParam(void* inImage, string name, char* xmlIn)
{
    int ret = 0;
    try
    {
        ResolveXml rXml;
        m_inImg = ((cv::Mat*)inImage)->clone();
        m_debugProcess = rXml.parseXmlInt(xmlIn, "Param", name, "debugProcess");
        m_debugResult = rXml.parseXmlInt(xmlIn, "Param", name, "debugResult");
        m_polygonAngle = rXml.parseXmlInt(xmlIn, "Param", name, "polygonAngle");
        ret = rXml.parseXmlRectROI(xmlIn, "Param", name, "rotRectROI", m_vecRectROI);
        if (ret != 0) return EC_resBaseParam;
        ret = rXml.parseXmlCircleROI(xmlIn, "Param", name, "circleROI", m_vecCircleROI);
        if (ret != 0) return EC_resBaseParam;
        ret = rXml.parseXmlPolygonROI(xmlIn, "Param", name, "polygonROI", m_vecPolygonROI);
        if (ret != 0) return EC_resBaseParam;
        ret = rXml.parseXmlPolygonROI(xmlIn, "Param", name, "delConROI", m_vecDelConROI);
        if (ret != 0) return EC_resBaseParam;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_resBaseParam;
    }
    return ret;
}

Rect parseBaseParam::pointToRect(const vector<cv::Point> inContours)
{
    cv::Rect outRect = Rect(0, 0, 0, 0);
    try
    {
        // 找到最小x和最大x
        int min_x = inContours[0].x;
        int max_x = inContours[0].x;
        for (int i = 1; i < inContours.size(); i++) {
            if (inContours[i].x < min_x) {
                min_x = inContours[i].x;
            }
            if (inContours[i].x > max_x) {
                max_x = inContours[i].x;
            }
        }
        // 找到最小y和最大y
        int min_y = inContours[0].y;
        int max_y = inContours[0].y;
        for (int i = 1; i < inContours.size(); i++) {
            if (inContours[i].y < min_y) {
                min_y = inContours[i].y;
            }
            if (inContours[i].y > max_y) {
                max_y = inContours[i].y;
            }
        }
        //构造所有点集外接矩形Rect
        outRect.x = min_x;
        outRect.y = min_y;
        outRect.width = abs(max_x - min_x);
        outRect.height = abs(max_y - min_y);
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
    }
    return outRect;
}

vector<cv::Point> parseBaseParam::rotatePolygon(vector<cv::Point> inContours, cv::Rect inRect, float rotateAngle)
{
    vector<cv::Point> outContours;
    try
    {
        //构造旋转中心
        cv::Point rotateCenter = cv::Point(0, 0);
        rotateCenter.x = inRect.x + inRect.width / 2;
        rotateCenter.y = inRect.y + inRect.height / 2;
        //多边形角点围绕旋转中心旋转
        double radian = -rotateAngle * CV_PI / 180.0;//角度转弧度,负号为统一方向性：顺时针为负，逆时针为正
        double cosVal = cos(radian);
        double sinVal = sin(radian);
        cv::Point outPoint = cv::Point(0, 0);
        for (size_t i = 0; i < inContours.size(); i++)
        {
            outPoint.x = (inContours[i].x - rotateCenter.x) * cosVal - (inContours[i].y - rotateCenter.y) * sinVal + rotateCenter.x;
            outPoint.y = (inContours[i].x - rotateCenter.x) * sinVal + (inContours[i].y - rotateCenter.y) * cosVal + rotateCenter.y;
            outContours.push_back(outPoint);
        }
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
    }
    return outContours;
}