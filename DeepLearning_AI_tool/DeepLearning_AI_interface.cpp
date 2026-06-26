#include "pch.h"
#include "../_common_files/FileLogEx.h"
#include "DeepLearning_AI_interface.h"
#include "DeepLearning_AI_detect.h"
#include "DeepLearning_AI_segment.h"
#include "DeepLearning_AI_obb.h"
#include "DeepLearning_AI_pose.h"
#include <mutex>

int DLE_AI_detect_init(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_detect_init";
	DeepLearning_AI_detect::configureAiLoggingForRequest(xmlIn, apiName);

	// 检查系统是否支持CUDA（GPU加速）
	if (torch::hasCUDA())
	{
		// 如果支持GPU，输出提示信息
		std::cout << "gpu" << std::endl;
	}

	int ret = 0;
	DeepLearning_AI_detect deepAI_detect;
	ret = deepAI_detect.initDetectAiObj(xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_detect(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_detect";
	DeepLearning_AI_detect::configureAiLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	// 执行目标检测操作
	DeepLearning_AI_detect deepAI_detect;
	ret = deepAI_detect.processOneImgByDetection(inImg, outImg, xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_detect_clean(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_detect_clean";
	DeepLearning_AI_detect::configureAiLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	DeepLearning_AI_detect deepAI_detect;
	ret = deepAI_detect.destoryDetectObject(xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_segment_init(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_segment_init";
	DeepLearning_AI_segment::configureAiLoggingForRequest(xmlIn, apiName);

	if (torch::hasCUDA())
	{
		std::cout << "gpu" << std::endl;
	}

	int ret = 0;
	DeepLearning_AI_segment deepAI_segment;
	ret = deepAI_segment.initSegmentAiObj(xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_segment(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_segment";
	DeepLearning_AI_segment::configureAiLoggingForRequest(xmlIn, apiName);

	int ret = 0;
	// 执行分割操作
	DeepLearning_AI_segment deepAI_segment;
	ret = deepAI_segment.processOneImgBySegment(inImg, outImg, xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_segment_clean(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_segment_clean";
	DeepLearning_AI_segment::configureAiLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	DeepLearning_AI_segment deepAI_segment;
	ret = deepAI_segment.destorySegmentObject(xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_obb_init(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_obb_init";
	DeepLearning_AI_obb::configureAiLoggingForRequest(xmlIn, apiName);

	if (torch::hasCUDA())
	{
		std::cout << "gpu" << std::endl;
	}

	int ret = 0;
	DeepLearning_AI_obb deepAI_obb;
	ret = deepAI_obb.initObbAiObj(xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_obb(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_obb";
	DeepLearning_AI_obb::configureAiLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	// 执行定向检测OBB操作
	DeepLearning_AI_obb deepAI_obb;
	ret = deepAI_obb.processOneImgByObb(inImg, outImg, xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_obb_clean(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_obb_clean";
	DeepLearning_AI_obb::configureAiLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	DeepLearning_AI_obb deepAI_obb;
	ret = deepAI_obb.destoryObbObject(xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_pose_init(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_pose_init";
	DeepLearning_AI_pose::configureAiLoggingForRequest(xmlIn, apiName);

	if (torch::hasCUDA())
	{
		std::cout << "gpu" << std::endl;
	}

	int ret = 0;
	DeepLearning_AI_pose deepAI_pose;
	ret = deepAI_pose.initPoseAiObj(xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_pose(void* inImg, void*& outImg, char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_pose";
	DeepLearning_AI_pose::configureAiLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	// 执行姿态关键点检测pose操作
	DeepLearning_AI_pose deepAI_pose;
	ret = deepAI_pose.processOneImgByPose(inImg, outImg, xmlIn, xmlOut, apiName);
	return ret;
}

int DLE_AI_pose_clean(char* xmlIn, char** xmlOut)
{
	std::string apiName = "DLE_AI_pose_clean";
	DeepLearning_AI_pose::configureAiLoggingForRequest(xmlIn, apiName);

	int ret = 0;

	DeepLearning_AI_pose deepAI_pose;
	ret = deepAI_pose.destoryPoseObject(xmlIn, xmlOut, apiName);
	return ret;
}
