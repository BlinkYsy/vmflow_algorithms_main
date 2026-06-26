#include "pch.h"
#include "DeepLearning_OCR_interface.h"
#include "DeepLearning_OCR.h"

int DLE_ocr_init(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_ocr_init";
	DeepLearning_OCR::configureOcrLoggingForRequest(xmlIn, apiName);

	// 检查系统是否支持CUDA（GPU加速）
	if (torch::hasCUDA())
	{
		// 如果支持GPU，输出提示信息
		std::cout << "gpu" << std::endl;
	}

	int ret = 0;
	DeepLearning_OCR deepAI_ocr;
	ret = deepAI_ocr.initOcrObj(xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_ocr(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_ocr";
	DeepLearning_OCR::configureOcrLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	// 执行OCR识别操作
	DeepLearning_OCR deepAI_ocr;
	ret = deepAI_ocr.processOneImgByOcr(inImg, outImg, xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_ocr_clean(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_ocr_clean";
	DeepLearning_OCR::configureOcrLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	DeepLearning_OCR deepAI_ocr;
	ret = deepAI_ocr.destoryOcrObject(xmlIn, xmlOut, apiName);
	return ret;
}
