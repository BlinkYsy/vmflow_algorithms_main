#include "pch.h"
#include "ColorProcess_interface.h"
#include "ColorProcess_color_extract.h"
#include "ColorProcess_color_transform.h"

int CPR_color_extract(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
	CExtColor extColor;
	int ret = extColor.cpr_extractColor(inImg, outImg, xmlIn, xmlOut);
	return ret;
}

int CPR_color_transform(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
	CTraColor traColor;
	int ret = traColor.cpr_transformColor(inImg, outImg, xmlIn, xmlOut);
	return ret;
}