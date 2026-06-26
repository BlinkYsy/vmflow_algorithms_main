#pragma once

//#ifndef __AFXWIN_H__
//	#error "在包含此文件之前包含 'pch.h' 以生成 PCH"
//#endif

#include "Resource.h"		// 主符号

#ifdef DEEPLEARNING_OCR_EXPORTS
#define DEEPLEARNING_OCR_API extern "C" __declspec(dllexport)
#else 
#define DEEPLEARNING_OCR_API extern "C" __declspec(dllimport)
#endif

extern "C"
{
	/*
	* 接口功能：	OCR识别
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_OCR_API int DLE_ocr(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	初始化 OCR识别模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_OCR_API int DLE_ocr_init(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	销毁 OCR识别模块
	* 开 发 者：	余斯谊
	* 传入参数：
	* 传出参数：
	*/
	DEEPLEARNING_OCR_API int DLE_ocr_clean(char* xmlIn, char** xmlOut);

}
