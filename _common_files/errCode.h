#pragma once
#ifndef _CPM_ERRORCODE_H
#define _CPM_ERRORCODE_H

enum ErrCode
{
	EC_SUCCESS = 0,											//成功
	EC_FAIL = -1,											//失败
	//上层接口交互(interface) 错误代码区间（20，150）
	EC_INTERFACE_CPM_dataTrans = 20,						//传入图像转换异常
	EC_INTERFACE_CPM_byteTrans = 21,						//传出图像转换异常
	EC_INTERFACE_CPM_interface = 22,						//算子接口异常
	EC_INTERFACE_CPM_delMem = 23,							//图像释放异常
	EC_INTERFACE_medInterface = 24,							//接口失败
	//基本参数解析 错误代码区间（150，200）
	EC_resBaseParam = 151,									//基本参数解析失败
	EC_iniBaseParamImg = 152,								//传入图像基本参数初始化失败
	//图像预处理 错误代码区间（200，600）
	//形态学处理 201-220
	EC_MORPHOLOGY_parseParam = 201,							//传入参数解析失败
	EC_MORPHOLOGY_interface = 202,							//函数接口异常
	EC_MORPHOLOGY_morphology_img = 203,						//morphology_img主函数异常
	EC_MORPHOLOGY_iniBaseParamImg = 204,					//传入图像基本参数初始化失败
	EC_MORPHOLOGY_morphologyImage = 205,					//图像形态学处理失败
	EC_MORPHOLOGY_morphModuleResult = 206,					//morphology模块结果输出失败
	EC_MORPHOLOGY_morphDilate = 207,						//膨胀失败
	EC_MORPHOLOGY_morphErode = 208,							//腐蚀失败
	EC_MORPHOLOGY_morphOpen = 209,							//开运算失败
	EC_MORPHOLOGY_morphClose = 210,							//闭运算失败
	EC_MORPHOLOGY_outputmorphXml = 211,						//传出参数构造失败
	//图像二值化 221-240
	EC_IMGBINARY_interface = 221,							//函数接口异常
	EC_IMGBINARY_parseParam = 222,							//传入参数解析失败
	EC_IMGBINARY_threshold_img = 223,						//threshold_img主函数异常
	EC_IMGBINARY_thresholdImage = 224,						//图像二值化处理失败
	EC_IMGBINARY_hardThreshold = 225,						//硬阈值二值化
	EC_IMGBINARY_thrModuleResult = 226,						//threshold模块结果输出失败
	EC_IMGBINARY_outputThrXml = 227,						//传出参数构造失败
	EC_IMGBINARY_adaptiveThreshold = 228,                    //自适应二值化
	EC_IMGBINARY_gaussianThreshold = 229,                    //高斯二值化
	EC_IMGBINARY_MeanThreshold = 230,
	//定位 错误代码区间（600，1000）
	// Blob工具 601-620
	EC_BLOB_interface = 601,								//函数接口异常
	EC_BLOB_parseParam = 602,								//传入参数解析失败
	EC_BLOB_outputBlobXml = 603,							//传出参数构造失败
	EC_BLOB_find_blob = 604,								//Blob主函数异常
	EC_BLOB_thresholdImg = 605,								//输入图像阈值化失败
	EC_BLOB_iniBaseParamImg = 606,							//传入图像基本参数初始化失败
	EC_BLOB_blobFindContours = 607,							//定位Blob失败
	EC_BLOB_blobModuleResult = 608,							//Blob模块结果输出失败
	EC_BLOB_doubleThreshold = 609,							//双阈值失败
	//快速匹配 621-640
	EC_MATCHFAST_makingTempFast = 621,						//制作模板失败
	EC_MATCHFAST_parseParamMakingTempFast = 622,			//制作模板传入参数解析失败
	EC_MATCHFAST_outputMakingTempFastXml = 623,				//制作模板传出参数构造失败
	EC_MATCHFAST_loadModelFast = 624,						//加载模板失败
	EC_MATCHFAST_interface = 625,							//模板匹配函数接口异常
	EC_MATCHFAST_parseParam = 626,							//模板匹配传入参数解析失败
	EC_MATCHFAST_match_fast = 627,							//match_fast主函数异常
	EC_MATCHFAST_outputMatchFastXml = 628,					//模板匹配传出参数构造失败
	EC_MATCHFAST_thresholdImg = 629,						//模板匹配输入图像阈值化失败
	EC_MATCHFAST_matchFastModuleResult = 630,				//快速匹配模块结果输出失败
	EC_MATCHFAST_destroyModelFast = 631,					//快速匹配销毁模板失败
	//顶点检测 641-660
	EC_PEAK_interface = 641,								//函数接口异常
	EC_PEAK_parseParam = 642,								//传入参数解析失败
	EC_PEAK_outputPeakXml = 643,							//传出参数构造失败
	EC_PEAK_findPeak = 644,									//findPeak主函数异常
	EC_PEAK_peakModuleResult = 645,							//peak模块结果输出失败
	EC_PEAK_filter = 646,									//降噪失败
	EC_PEAK_selectPeak = 647,								//顶点筛选失败
	//颜色处理 错误代码区间（1000，1200）
	//颜色抽取 1001-1020
	EC_EXTCOLOR_interface = 1001,							//函数接口异常
	EC_EXTCOLOR_parseParam = 1002,							//传入参数解析失败
	EC_EXTCOLOR_extractColor = 1003,						//extractColor主函数异常
	EC_EXTCOLOR_extColor = 1004,							//颜色抽取失败
	EC_EXTCOLOR_extColorModuleResult = 1005,				//extColor模块结果输出失败
	EC_EXTCOLOR_areaContours = 1006,						//计算颜色总面积异常
	//颜色转换 1021-1040
	EC_TRACOLOR_interface = 1021,							//函数接口异常
	EC_TRACOLOR_parseParam = 1022,							//传入参数解析失败
	EC_TRACOLOR_transformColor = 1023,						//transformColor主函数异常
	EC_TRACOLOR_traColor = 1024,							//颜色转换失败
	EC_TRACOLOR_traColorModuleResult = 1025,				//traColor模块结果输出失败
	EC_TRACOLOR_outputTransformColorXml = 1026,				//传出参数构造失败
	EC_TRACOLOR_traRGB2YUV = 1027,							//RGB2YUV失败
	//测量工具 错误代码区间（1200，1400）
	//点线测量 1201-1220
	EC_P2LMEASURE_interface = 1201,							//函数接口异常
	//定制算子 错误代码区间（1400，2000）
	//I-004 针扣、弹簧、插针检测 1401-1420
	EC_NEEDLE2SPRING_interface = 1401,						//函数接口异常
	EC_TUBEHEAD_parseParam = 1402,							//传入参数解析失败
	EC_TUBEHEAD_tubeHead = 1403,							//tubeHead主函数异常
	EC_TUBEHEAD_locTubeHead = 1404,							//软管头部检测失败
	EC_TUBEHEAD_tubeHeadModuleResult = 1405,				//tubeHead模块结果输出失败
	EC_TUBEHEAD_outputTubeHeadXml = 1406,					//传出参数构造失败
	EC_TUBEHEAD_extTubeHead = 1407,							//提取软管头部失败
	EC_TUBEHEAD_locPoints = 1408,							//定位软管顶点失败
	EC_TUBEHEAD_sortPoints = 1409,							//排序点集错误
	//缺陷检测 错误代码区间（2000，2500）
	//直线边缘缺陷检测 2001-2020
	EC_LINEEDGEINSPECT_interface = 2001,					//函数接口异常
	EC_LINEEDGEINSPECT_parseParam = 2002,					//传入参数解析失败
	EC_LINEEDGEINSPECT_lineEdgeInspect = 2003,				//lineEdgeInspect主函数异常
	EC_LINEEDGEINSPECT_lineEdgeInsp = 2004,					//直线边缘缺陷检测失败
	EC_LINEEDGEINSPECT_lineEdgeInspModuleResult = 2005,		//lineEdgeInspect模块结果输出失败
	EC_LINEEDGEINSPECT_delNoise = 2006						//降噪失败
};
#endif