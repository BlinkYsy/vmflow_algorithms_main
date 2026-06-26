#pragma once

//#ifndef __AFXWIN_H__
//	#error "在包含此文件之前包含 'pch.h' 以生成 PCH"
//#endif

#include "Resource.h"		// 主符号

#ifdef DEEPLEARNING_AI_EXPORTS
#define DEEPLEARNING_API extern "C" __declspec(dllexport)
#else 
#define DEEPLEARNING_API extern "C" __declspec(dllimport)
#endif

extern "C"
{
	/*
	* 接口功能：	AI_目标检测
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_detect(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	AI_实例分割
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_segment(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	AI_旋转目标检测
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_obb(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	AI_姿态识别
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_pose(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	初始化 AI_目标检测模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_detect_init(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	初始化 AI_实例分割模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_segment_init(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	初始化 AI_旋转目标检测模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_obb_init(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	初始化 AI_实例分割模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_pose_init(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	销毁 AI_目标检测模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_detect_clean(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	销毁 AI_实例分割模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_segment_clean(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	销毁 AI_旋转目标检测模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_obb_clean(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	销毁 AI_实例分割模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_API int DLE_AI_pose_clean(char* xmlIn, char** xmlOut);

}
