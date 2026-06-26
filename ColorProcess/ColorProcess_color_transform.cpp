#include "pch.h"
#include "ColorProcess_color_transform.h"
#include "../_common_files/errCode.h"
#include "../_common_files/ResolveXml.h"

int CTraColor::parseParam(void* inImg, char* xmlIn)
{
    int ret = 0;
    try
    {
        //基本参数解析
        ret = resBaseParam(inImg, "CPR_color_transform", xmlIn);
        if (ret != 0) return ret;

        ResolveXml rXml;
        m_colorTransformType = rXml.parseXmlInt(xmlIn, "Param", "CPR_color_transform", "colorTransformType");
        m_showChannel = rXml.parseXmlInt(xmlIn, "Param", "CPR_color_transform", "showChannel");

        if (m_debugProcess || m_debugResult)
        {
            CreateDirectory("d:\\debug", 0);
            CreateDirectory("d:\\debug\\extractColor", 0);
        }
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_parseParam;
    }
    return ret;
}

//Y：亮度信息，黑白图主要保留的部分，（0.299R + 0.587G + 0.114B按人眼对绿色更敏感的特点加权，更接近感知亮度）
//U、V：两路色度信息，用来描述颜色偏向
//分离亮度 Y 与色度 U/V。Y 通道适合只看明暗结构
int CTraColor::traRGB2YUV(cv::Mat inImg, cv::Mat& outImg)
{
    int ret = 0;
    try
    {
        if (inImg.channels() != 3)
            return -1;

        cv::Mat imageYUV;
        cv::cvtColor(inImg, imageYUV, COLOR_RGB2YUV);
        std::vector<Mat> mv;
        split(imageYUV, (vector<Mat>&)mv);
        switch (m_showChannel)
        {
        case showChannel_FirstChannel:
            outImg = mv[0].clone();
            break;
        case showChannel_SecondChannel:
            outImg = mv[1].clone();
            break;
        case showChannel_ThirdChannel:
            outImg = mv[2].clone();
            break;
        default:
            break;
        }
        //cv::cvtColor(image, outImg, COLOR_GRAY2BGR);
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_traRGB2YUV;
    }
    return ret;
}

//H：色相，S：饱和度，I：亮度 （I = (R + G + B) / 3）
//色相、饱和度和亮度拆开，适合做颜色分析
//只输出 Y、U、V 中指定的一个单通道图。
int CTraColor::traRGB2HSI(cv::Mat inImg, cv::Mat& outImg)
{
    int ret = 0;
    try
    {
        outImg = Mat(inImg.rows, inImg.cols, inImg.type());
        if (inImg.channels() != 3)
            return -1;

        float r, g, b, H, S, I, num, den, theta, sum, min_RGB;
        for (int i = 0; i < inImg.rows; i++)
        {
            for (int j = 0; j < inImg.cols; j++)
            {
                b = inImg.at<Vec3b>(i, j)[0];
                g = inImg.at<Vec3b>(i, j)[1];
                r = inImg.at<Vec3b>(i, j)[2];

                // 归一化
                b = b / 255.0;
                g = g / 255.0;
                r = r / 255.0;

                num = 0.5 * ((r - g) + (r - b));
                den = sqrt((r - g) * (r - g) + (r - b) * (g - b));
                theta = acos(num / den);
                if (den == 0)
                    H = 0; // 分母不能为0
                else
                    if (b <= g)
                        H = theta;
                    else
                        H = (2 * PI - theta);
                min_RGB = min(min(b, g), r); // min(R,G,B)
                sum = b + g + r;
                if (sum == 0)
                    S = 0;
                else
                    S = 1 - 3 * min_RGB / sum;
                I = sum / 3.0;
                H = H / (2 * PI);
                // 将S分量和H分量都扩充到[0,255]区间以便于显示;一般H分量在[0,2pi]之间，S在[0,1]之间



                switch (m_showChannel)
                {
                case showChannel_FirstChannel:
                    outImg.at<Vec3b>(i, j)[0] = H * 255;
                    outImg.at<Vec3b>(i, j)[1] = H * 255;
                    outImg.at<Vec3b>(i, j)[2] = H * 255;

                    break;
                case showChannel_SecondChannel:
                    outImg.at<Vec3b>(i, j)[1] = S * 255;
                    outImg.at<Vec3b>(i, j)[0] = S * 255;
                    outImg.at<Vec3b>(i, j)[2] = S * 255;
                    break;
                case showChannel_ThirdChannel:
                    outImg.at<Vec3b>(i, j)[2] = I * 255;
                    outImg.at<Vec3b>(i, j)[0] = I * 255;
                    outImg.at<Vec3b>(i, j)[1] = I * 255;

                    break;
                default:
                    break;
                }
            }
        }


    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_traRGB2YUV;
    }
    return ret;
}

//H：色相，颜色类别，如红、绿、蓝  
//S：饱和度，颜色“鲜不鲜艳”
//V：明亮度，颜色“亮不亮” V = max(R, G, B)
//将颜色和亮度分开。常用于按颜色提取目标
//只输出 H、S、V 中由 m_showChannel 指定的一个单通道图
int CTraColor::traRGB2HSV(cv::Mat inImg, cv::Mat& outImg)
{
    int ret = 0;
    try
    {
        if (inImg.channels() != 3)
            return -1;
        cv::Mat imageHSV;
        cv::cvtColor(inImg, imageHSV, COLOR_RGB2HSV);
        std::vector<Mat> mv;
        split(imageHSV, (vector<Mat>&)mv);
        switch (m_showChannel)
        {
        case showChannel_FirstChannel:
            outImg = mv[0].clone();
            break;
        case showChannel_SecondChannel:
            outImg = mv[1].clone();
            break;
        case showChannel_ThirdChannel:
            outImg = mv[2].clone();
            break;
        default:
            break;
        }
        //cv::cvtColor(image, outImg, COLOR_GRAY2BGR);
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_traRGB2YUV;
    }
    return ret;
}

//输出灰度单通道图
int CTraColor::traRGB2GRAY(cv::Mat inImg, cv::Mat& outImg)
{
    int ret = 0;
    try
    {
        if (inImg.channels() != 3)
            return -1;
        cv::Mat imageGRAY;
        //OpenCV标准加权亮度转换
        cv::cvtColor(inImg, imageGRAY, COLOR_RGB2GRAY);
        std::vector<Mat> mv;
        split(imageGRAY, (vector<Mat>&)mv);
        switch (m_showChannel)
        {
        case showChannel_FirstChannel:
            outImg = mv[0].clone();
            break;
        case showChannel_SecondChannel:
            outImg = mv[0].clone();
            break;
        case showChannel_ThirdChannel:
            outImg = mv[0].clone();
            break;
        default:
            break;
        }
        //cv::cvtColor(image, outImg, COLOR_GRAY2BGR);
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_traRGB2YUV;
    }
    return ret;
}

int CTraColor::traColor(cv::Mat inImg, cv::Mat& outImg)
{
    int ret = 0;
    try
    {
        //颜色转换
        switch (m_colorTransformType)
        {
        case colorTransformType_RGB2GRAY:
            traRGB2GRAY(inImg, outImg);
            break;
        case colorTransformType_RGB2HSV:
            traRGB2HSV(inImg, outImg);
            break;
        case colorTransformType_RGB2HSI:
            traRGB2HSI(inImg, outImg);
            break;
        case colorTransformType_RGB2YUV:
            traRGB2YUV(inImg, outImg);
            break;
        default:
            break;
        }
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_traColor;
    }
    return ret;
}

int	CTraColor::traColorModuleResult(cv::Mat inImg, void*& outImg)
{
    int ret = 0;
    try
    {
        //输出图像构造
        m_renderparam.conOutImg(inImg, outImg);

        m_moduStatus = 1;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_traColorModuleResult;
    }
    return ret;
}

int	CTraColor::transformColor(void*& outImg)
{
    int ret = 0;
    try
    {
        //输入图像基本参数初始化
        ret = iniBaseParamImg(m_inImg, m_iniImg);
        if (ret != 0) return ret;

        //颜色转换
        ret = traColor(m_iniImg, m_traColorImg);
        if (ret != 0) return ret;

        //模块结果输出
        ret = traColorModuleResult(m_traColorImg, outImg);
        if (ret != 0) return ret;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_transformColor;
    }
    return ret;
}

int	CTraColor::outputTransformColorXml(char* xmlOut, string funName)
{
    int ret = 0;
    try
    {
        // 构建XML字符串
        CMarkup xml;
        xml.SetDoc("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n");
        xml.AddElem((MCD_CSTR)"Param");
        xml.IntoElem();

        xml.AddElem((MCD_CSTR)funName.c_str());
        xml.IntoElem();

        CString str;

        CString OUT_INT = "%d";

        //模块状态
        str.Format(OUT_INT, m_moduStatus);
        xml.AddElem((MCD_CSTR)"moduStatus", (MCD_CSTR)str.GetBuffer());
        xml.SetAttrib("CH", "模块状态");

        xml.OutOfElem();

        CString Out = xml.GetDoc();
        strcpy(xmlOut, (char*)Out.GetBuffer());
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_outputTransformColorXml;
    }
    return ret;
}

int CTraColor::cpr_transformColor(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
    int ret = 0;
    try
    {
        //解析传入参数
        ret = parseParam(inImg, xmlIn);
        if (ret != 0)  return ret;

        //transformColor主函数
        ret = transformColor(outImg);
        if (ret != 0)  return ret;

        //构造传出参数
        ret = outputTransformColorXml(*xmlOut, "CPR_color_transform");
        if (ret != 0)  return ret;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_TRACOLOR_interface;
    }
    return ret;
}