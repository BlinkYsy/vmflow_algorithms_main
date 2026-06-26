#pragma once
#include "Resource.h"

#ifdef DEEPAI_EXPORTS
#define DEEPAI_API extern "C" __declspec(dllexport)
#else 
#define DEEPAI_API extern "C" __declspec(dllimport)
#endif

extern "C"
{
	/*
	* 接口功能：	
	* 开 发 者：	
	* 传入参数：
	* 传出参数：
	*/
	DEEPAI_API int AI_seg(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
	DEEPAI_API int AI_class(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
	DEEPAI_API int AI_detect(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);

	DEEPAI_API int AI_initSeg(char* xmlIn, char** xmlOut);
	DEEPAI_API int AI_initClass(char* xmlIn, char** xmlOut);
	DEEPAI_API int AI_initDetect(char* xmlIn, char** xmlOut);

	DEEPAI_API int AI_segDes(char* xmlIn, char** xmlOut);
}