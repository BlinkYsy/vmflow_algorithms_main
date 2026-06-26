#pragma once

//#ifndef __AFXWIN_H__
//	#error "在包含此文件之前包含 'pch.h' 以生成 PCH"
//#endif

#include "resource.h"		// 主符号

#ifdef COLORPROCESS_EXPORTS
#define COLORPROCESS_API extern "C" __declspec(dllexport)
#else 
#define COLORPROCESS_API extern "C" __declspec(dllimport)
#endif

extern "C"
{
	/*
	* 接口功能：	颜色抽取
	* 开 发 者：	谭风
	* 传入参数：
	* 传出参数：
	*/
	COLORPROCESS_API int CPR_color_extract(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	颜色转换
	* 开 发 者：	谭风
	* 传入参数：
	* 传出参数：
	*/
	COLORPROCESS_API int CPR_color_transform(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
}