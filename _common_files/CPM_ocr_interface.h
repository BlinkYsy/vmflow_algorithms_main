// CPM_ocr.h: CPM_ocr DLL 的主标头文件
//

#pragma once

//#ifndef __afxwin_h__
//	#error "在包含此文件之前包含 'pch.h' 以生成 pch"
//#endif

#include "resource.h"		// 主符号

#ifdef CPMOCR_EXPORTS
#define CPMOCR_API extern "C" __declspec(dllexport)
#else 
#define CPMOCR_API extern "C" __declspec(dllimport)
#endif

extern "C"
{
	CPMOCR_API int OCR_init(char* xmlIn, char** xmlOut);

	/*
	* 接口功能：	文字识别
	* 开 发 者：	冷斌
	* 传入参数：
	* 传出参数：
	*/
	CPMOCR_API int OCR_scan(void* inImg, char* xmlIn, char** xmlOut);

	CPMOCR_API int OCR_des(char* xmlIn, char** xmlOut);
}
