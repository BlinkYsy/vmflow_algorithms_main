#include "pch.h"
#include "ColorProcess_color_extract.h"
#include "../_common_files/errCode.h"
#include "../_common_files/ResolveXml.h"

/*
    解析颜色抽取模块的运行参数，包括颜色空间、颜色反转标志以及三个通道的上下阈值。
*/
int CExtColor::parseRunParam(void* inImg, char* xmlIn)
{
    int ret = 0;
    try
    {
        string name = "CPR_color_extract";
        ResolveXml rXml;
        m_runparam.colorSpace = rXml.parseXmlInt(xmlIn, "Param", name, "colorSpace");
        istringstream(rXml.parseXmlStr(xmlIn, "Param", name, "colorReversal")) >> boolalpha >> m_runparam.colorReversal;
        m_runparam.channelLow_1 = rXml.parseXmlInt(xmlIn, "Param", name, "channelLow_1");
        m_runparam.channelHigh_1 = rXml.parseXmlInt(xmlIn, "Param", name, "channelHigh_1");
        m_runparam.channelLow_2 = rXml.parseXmlInt(xmlIn, "Param", name, "channelLow_2");
        m_runparam.channelHigh_2 = rXml.parseXmlInt(xmlIn, "Param", name, "channelHigh_2");
        m_runparam.channelLow_3 = rXml.parseXmlInt(xmlIn, "Param", name, "channelLow_3");
        m_runparam.channelHigh_3 = rXml.parseXmlInt(xmlIn, "Param", name, "channelHigh_3");
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_EXTCOLOR_parseParam;
    }
    return ret;
}

/*
    解析颜色抽取模块的结果显示参数，当前预留为空实现，便于后续扩展结果展示相关配置。
*/
int CExtColor::parseResultParam(void* inImg, char* xmlIn)
{
    int ret = 0;
    try
    {
        string name = "CPR_color_extract";
        ResolveXml rXml;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_EXTCOLOR_parseParam;
    }
    return ret;
}

/*
    统一解析颜色抽取模块所需的全部输入参数，包括基础参数、运行参数和结果参数，同时在调试模式下创建调试输出目录。
*/
int CExtColor::parseParam(void* inImg, char* xmlIn)
{
    int ret = 0;
    try
    {
        // 基本参数解析
        ret = resBaseParam(inImg, "CPR_color_extract", xmlIn);
        if (ret != 0) return ret;

        // 运行参数解析
        ret = parseRunParam(inImg, xmlIn);
        if (ret != 0) return ret;

        // 结果显示参数解析
        ret = parseResultParam(inImg, xmlIn);
        if (ret != 0) return ret;

        if (m_debugProcess || m_debugResult)
        {
            CreateDirectory("d:\\debug", 0);
            CreateDirectory("d:\\debug\\extractColor", 0);
        }
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_EXTCOLOR_parseParam;
    }
    return ret;
}

/*
    统计输入二值图像中非零像素的数量，并将统计结果写入颜色总面积。
*/
int CExtColor::areaContours(cv::Mat inImg)
{
    int ret = 0;
    try
    {
        int counter = 0;
        for (int m = 0; m < inImg.rows; m++)
        {
            uchar* p_inImg = inImg.ptr<uchar>(m); // 获取m行的首地址
            for (int n = 0; n < inImg.cols; n++)
            {
                if (p_inImg[n] > 0)
                    counter++;
            }
        }

        //// 迭代器访问像素点
        //Mat_<uchar>::iterator it = inImg.begin<uchar>();
        //Mat_<uchar>::iterator itend = inImg.end<uchar>();
        //for (; it != itend; ++it)
        //{
        //    if ((*it) > 0) counter += 1;
        //}

        m_moduleresult.colorTotalArea = counter;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_EXTCOLOR_areaContours;
    }
    return ret;
}

/*
    将输入的 BGR 图像逐像素转换为 HSI 颜色空间，并输出 H、S、I 三个分量。
*/
int CExtColor::rgbToHSI(cv::Mat inImg, cv::Mat& outImg)
{
    try
    {
        for (int y = 0; y < inImg.rows; ++y)
        {
            for (int x = 0; x < inImg.cols; ++x)
            {
                Vec3b rgbPixel = inImg.at<Vec3b>(y, x);
                float R = rgbPixel[2] / 255.0; // 红色通道
                float G = rgbPixel[1] / 255.0; // 绿色通道
                float B = rgbPixel[0] / 255.0; // 蓝色通道

                // 计算亮度 I
                float I = (R + G + B) / 3.0;

                // 计算饱和度 S
                float minRGB = min(R, min(G, B));
                float S = (I == 0) ? 0 : 1 - (minRGB / I);

                // 计算色相 H
                float H;
                if (S == 0)
                {
                    H = 0; // 如果饱和度为0，色相为0
                }
                else
                {
                    float num = 0.5 * ((R - G) + (R - B));
                    float den = sqrt((R - G) * (R - G) + (R - B) * (G - B));
                    float theta = acos(num / (den + 1e-10)); // 防止除以零
                    if (B <= G)
                    {
                        H = theta;
                    }
                    else
                    {
                        H = 2 * CV_PI - theta;
                    }
                    H = H * (180.0 / CV_PI); // 转换为度
                }

                // 将 HSI 值存储在 hsiImage 中
                outImg.at<Vec3f>(y, x) = Vec3f(H, S, I);
            }
        }
    }
    catch (const std::exception&)
    {
        return -1;
    }
    return 0;
}

/*
    按照指定颜色空间和阈值范围执行颜色抽取，生成掩膜图，并根据配置决定是否进行颜色反转及面积统计。
*/
int CExtColor::extColor(cv::Mat inImg, cv::Mat& outImg)
{
    int ret = 0;
    if (inImg.size == 0)
        return -1;
    try
    {
        // 颜色抽取
        cv::Mat img;
        cv::Mat hsvImg;
        cv::Mat inrImg;
        if (m_runparam.colorSpace == 0) // RGB抽取
        {
            inrImg = Mat::zeros(Size(inImg.cols, inImg.rows), CV_8UC1);
            for (int y = 0; y < inImg.rows; ++y)
            {
                for (int x = 0; x < inImg.cols; ++x)
                {
                    Vec3b pixel = inImg.at<Vec3b>(y, x);
                    uchar blue = pixel[0];
                    uchar green = pixel[1];
                    uchar red = pixel[2];

                    // 判断 RGB 值是否在范围内
                    if (red >= m_runparam.channelLow_1 && red <= m_runparam.channelHigh_1 &&
                        green >= m_runparam.channelLow_2 && green <= m_runparam.channelHigh_2 &&
                        blue >= m_runparam.channelLow_3 && blue <= m_runparam.channelHigh_3)
                    {
                        // 将符合条件的像素变为白色
                        inrImg.at<uchar>(y, x) = 255; // 设置为白色
                    }
                    else
                    {
                        // 可选：将不符合条件的像素设置为黑色（默认为0）
                        inrImg.at<uchar>(y, x) = 0; // 设置为黑色
                    }
                }
            }
        }
        if (m_runparam.colorSpace == 1) // HSV抽取
        {
            if (inImg.channels() == 3)
                cv::cvtColor(inImg, hsvImg, COLOR_BGR2HSV);
            else
                hsvImg = inImg;

            //三通道阈值筛选
            cv::inRange(
                hsvImg,
                Scalar(m_runparam.channelLow_1, m_runparam.channelLow_2, m_runparam.channelLow_3),
                Scalar(m_runparam.channelHigh_1, m_runparam.channelHigh_2, m_runparam.channelHigh_3),
                inrImg
            );
        }
        if (m_runparam.colorSpace == 2) // HSI抽取
        {
            inrImg = Mat::zeros(Size(inImg.cols, inImg.rows), CV_32FC3);
            ret = rgbToHSI(inImg, inrImg);
        }

        int intColorReversal = m_runparam.colorReversal ? 1 : 0;
        switch (intColorReversal)
        {
        case 0:
            // 颜色总面积
            ret = areaContours(inrImg);
            if (ret != 0) return ret;
            outImg = inrImg;
            break;
        case 1:
            // 颜色反转
            bitwise_not(inrImg, outImg);
            // 颜色总面积
            areaContours(outImg);
            if (ret != 0) return ret;
            break;
        default:
            return EC_EXTCOLOR_extColor;
        }
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_EXTCOLOR_extColor;
    }
    return ret;
}

/*
    构造颜色抽取模块的图像输出结果，并更新模块执行状态。
*/
int CExtColor::extColorModuleResult(cv::Mat inImg, void*& outImg)
{
    int ret = 0;
    try
    {
        // 输出图像构造
        m_renderparam.conOutImg(inImg, outImg);

        // 模块状态
        m_moduleresult.moduStatus = true;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_EXTCOLOR_extColorModuleResult;
    }
    return ret;
}

/*
    执行颜色抽取模块的完整处理流程，包括输入图像基本参数初始化、颜色抽取以及模块结果输出。
*/
int CExtColor::extractColor(void*& outImg)
{
    int ret = 0;
    try
    {
        // 输入图像基本参数初始化
        ret = iniBaseParamImg(m_inImg, m_iniImg);
        if (ret != 0) return ret;

        // 颜色抽取
        ret = extColor(m_iniImg, m_extColorImg);
        if (ret != 0) return ret;

        // 模块结果输出
        ret = extColorModuleResult(m_extColorImg, outImg);
        if (ret != 0) return ret;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_EXTCOLOR_extractColor;
    }
    return ret;
}

/*
    将颜色抽取模块的执行结果组织为 XML 字符串输出，主要包含模块状态和颜色总面积。
*/
int CExtColor::outputExtractColorXml(char* xmlOut, string funName)
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

        // 模块状态
        int moduStatus = m_moduleresult.moduStatus ? 1 : 0;
        str.Format(OUT_INT, moduStatus);
        xml.AddElem((MCD_CSTR)"moduStatus", (MCD_CSTR)str.GetBuffer());
        xml.SetAttrib("CH", "模块状态");

        // 颜色总面积
        str.Format(OUT_INT, m_moduleresult.colorTotalArea);
        xml.AddElem((MCD_CSTR)"colorTotalArea", (MCD_CSTR)str.GetBuffer());
        xml.SetAttrib("CH", "颜色总面积");

        xml.OutOfElem();

        CString Out = xml.GetDoc();
        strcpy(xmlOut, (char*)Out.GetBuffer());
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_MORPHOLOGY_outputmorphXml;
    }
    return ret;
}

/*
    颜色抽取模块的对外总入口，负责串联参数解析、主处理流程和 XML 结果输出。
*/
int CExtColor::cpr_extractColor(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
    int ret = 0;
    try
    {
        // 解析传入参数
        ret = parseParam(inImg, xmlIn);
        if (ret != 0) return ret;

        // extractColor主函数
        ret = extractColor(outImg);
        if (ret != 0) return ret;

        // 构造传出参数
        ret = outputExtractColorXml(*xmlOut, "CPR_color_extract");
        if (ret != 0) return ret;
    }
    catch (cv::Exception& e)
    {
        printf("cv::Ex %s\n", e.what());
        return EC_EXTCOLOR_interface;
    }
    return ret;
}
