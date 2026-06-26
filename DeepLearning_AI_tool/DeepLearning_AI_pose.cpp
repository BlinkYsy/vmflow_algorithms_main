#include "pch.h"
#include "DeepLearning_AI_pose.h"
#include "../_common_files/FileLogEx.h"
#include "../_common_files/ResolveXml.h"
#include "../_common_files/renderParam.h"
// 推理与预处理所需的标准库/第三方库
#include <iostream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cctype>
#include <cmath>
#include <limits>

// 全局模型对象映射表：moduleID → DeepLearning_AI_pose 实例指针
// DLL 内所有推理流程通过此表查找已注册的模型对象
std::unordered_map<int, DeepLearning_AI_pose*> g_mapDLE_AI_pose;

// 全局日志对象，用于记录所有操作日志
extern CFileLogEx g_log;
extern std::once_flag g_aiLoggerInitFlag;

namespace
{
	thread_local bool poseAiLoggingEnabled = false;

	void ensurePoseAiLoggerInitialized()
	{
		if (!poseAiLoggingEnabled)
			return;

		std::call_once(g_aiLoggerInitFlag, []()
			{
				char logPath[] = "d:\\debug\\log";
				g_log.initLog(logPath, "DeepLearning_AI_pose.log");
				g_log.Open(logPath, "DeepLearning_AI_pose.log");
			});
	}

	bool tryParseTensorSizeMismatch(const std::string& message, int& tensorSizeA, int& tensorSizeB)
	{
		const std::string prefixA = "The size of tensor a (";
		const std::string middle = ") must match the size of tensor b (";
		const size_t startA = message.find(prefixA);
		if (startA == std::string::npos)
			return false;

		const size_t valueAStart = startA + prefixA.size();
		const size_t middlePos = message.find(middle, valueAStart);
		if (middlePos == std::string::npos)
			return false;

		const size_t valueBStart = middlePos + middle.size();
		const size_t valueBEnd = message.find(')', valueBStart);
		if (valueBEnd == std::string::npos)
			return false;

		try
		{
			tensorSizeA = std::stoi(message.substr(valueAStart, middlePos - valueAStart));
			tensorSizeB = std::stoi(message.substr(valueBStart, valueBEnd - valueBStart));
			return tensorSizeA > 0 && tensorSizeB > 0;
		}
		catch (...)
		{
			return false;
		}
	}

	int estimateYoloGridCount(int inputSize)
	{
		if (inputSize <= 0)
			return 0;

		const int stride8 = inputSize / 8;
		const int stride16 = inputSize / 16;
		const int stride32 = inputSize / 32;
		return stride8 * stride8 + stride16 * stride16 + stride32 * stride32;
	}

	int suggestInputSizeFromTensorMismatch(int currentInputSize, int tensorSizeA, int tensorSizeB)
	{
		const int currentGridCount = estimateYoloGridCount(currentInputSize);
		if (currentGridCount <= 0)
			return 0;

		double scale = 0.0;
		if (tensorSizeA == currentGridCount)
		{
			scale = std::sqrt(static_cast<double>(tensorSizeB) / static_cast<double>(tensorSizeA));
		}
		else if (tensorSizeB == currentGridCount)
		{
			scale = std::sqrt(static_cast<double>(tensorSizeA) / static_cast<double>(tensorSizeB));
		}
		else
		{
			const int larger = std::max(tensorSizeA, tensorSizeB);
			const int smaller = std::min(tensorSizeA, tensorSizeB);
			scale = std::sqrt(static_cast<double>(larger) / static_cast<double>(smaller));
		}

		if (!(scale > 0.0) || !std::isfinite(scale))
			return 0;

		const double suggested = static_cast<double>(currentInputSize) * scale;
		const int rounded = static_cast<int>(std::lround(suggested));
		if (rounded <= 0)
			return 0;

		const int roundedGridCount = estimateYoloGridCount(rounded);
		if (roundedGridCount != tensorSizeA && roundedGridCount != tensorSizeB)
			return 0;

		return rounded;
	}

	std::string buildInputSizeMismatchHint(const std::string& message, int currentInputSize)
	{
		int tensorSizeA = 0;
		int tensorSizeB = 0;
		if (!tryParseTensorSizeMismatch(message, tensorSizeA, tensorSizeB))
			return std::string();

		const int suggestedInputSize = suggestInputSizeFromTensorMismatch(currentInputSize, tensorSizeA, tensorSizeB);
		if (suggestedInputSize <= 0 || suggestedInputSize == currentInputSize)
			return std::string();

		std::string hint = " Possible inputSize mismatch: current inputSize=";
		hint.append(std::to_string(currentInputSize));
		hint.append(", but this TorchScript pose model appears to expect inputSize=");
		hint.append(std::to_string(suggestedInputSize));
		hint.append(".");
		return hint;
	}

	int getFixedSquareOnnxInputSize(const Ort::Session& session)
	{
		if (session.GetInputCount() == 0)
			return 0;

		auto inputTypeInfo = session.GetInputTypeInfo(0);
		auto tensorTypeInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		const std::vector<int64_t> inputShape = tensorTypeInfo.GetShape();
		if (inputShape.size() < 4)
			return 0;

		const int64_t inputHeight = inputShape[2];
		const int64_t inputWidth = inputShape[3];
		if (inputHeight <= 0 || inputWidth <= 0)
			return 0;
		if (inputHeight != inputWidth)
			return 0;
		if (inputHeight > static_cast<int64_t>(std::numeric_limits<int>::max()))
			return 0;

		return static_cast<int>(inputHeight);
	}

	std::string buildOnnxInputSizeHint(int currentInputSize, int expectedInputSize)
	{
		if (currentInputSize <= 0 || expectedInputSize <= 0 || currentInputSize == expectedInputSize)
			return std::string();

		std::string hint = "ONNX model expects inputSize=";
		hint.append(std::to_string(expectedInputSize));
		hint.append(", but current inputSize=");
		hint.append(std::to_string(currentInputSize));
		hint.append(". Please update the XML parameter <inputSize> to match the model export size.");
		return hint;
	}
}

void DeepLearning_AI_pose::configureAiLoggingForRequest(const char* xmlIn, const std::string& apiName)
{
	poseAiLoggingEnabled = false;
	if (xmlIn == nullptr)
		return;

	ResolveXml xml;
	const int debugProcess = xml.parseXmlInt(xmlIn, "Param", apiName, "debugProcess");
	const int debugResult = xml.parseXmlInt(xmlIn, "Param", apiName, "debugResult");
	poseAiLoggingEnabled = debugProcess == 1 || debugResult == 1;
}

bool DeepLearning_AI_pose::addAiLog(long logTypeCode, const char* content)
{
	if (!poseAiLoggingEnabled || content == nullptr)
		return false;

	ensurePoseAiLoggerInitialized();
	return g_log.AddLog(logTypeCode, content);
}

namespace
{
	std::vector<int> parseCategoryIds(const std::string& value);
	cv::Scalar getColorForCategory(int category);
	std::string getCategoryDisplayName(int category);
	cv::Mat renderPoseImage(
		const cv::Mat& srcImg,
		const std::vector<cv::RotatedRect>& rects,
		const std::vector<int>& categories,
		const std::vector<cv::Point2f>& keypoints);

	std::vector<int> parseCategoryIds(const std::string& value)
	{
		std::vector<int> ids;
		std::stringstream ss(value);
		std::string token;
		while (std::getline(ss, token, ','))
		{
			if (token.empty())
				continue;
			ids.push_back(std::atoi(token.c_str()));
		}
		return ids;
	}

	cv::Scalar getColorForCategory(int category)
	{
		static const std::vector<cv::Scalar> palette = {
			cv::Scalar(0, 255, 0),
			cv::Scalar(0, 0, 255),
			cv::Scalar(255, 0, 0),
			cv::Scalar(0, 255, 255),
			cv::Scalar(255, 255, 0),
			cv::Scalar(255, 0, 255),
			cv::Scalar(0, 128, 255),
			cv::Scalar(128, 0, 255),
			cv::Scalar(255, 128, 0),
			cv::Scalar(0, 200, 120)
		};

		if (category < 0)
			return cv::Scalar(200, 200, 200);

		return palette[static_cast<size_t>(category) % palette.size()];
	}

	std::string getCategoryDisplayName(int category)
	{
		if (category < 0)
			return "cls_unknown";
		return "cls:" + std::to_string(category);
	}

	cv::Mat renderPoseImage(
		const cv::Mat& srcImg,
		const std::vector<cv::RotatedRect>& rects,
		const std::vector<int>& categories,
		const std::vector<cv::Point2f>& keypoints)
	{
		cv::Mat vis = srcImg.clone();
		const bool hasCategoryPerBox = categories.size() == rects.size();
		int keypointsPerBox = 0;
		if (!keypoints.empty() && !rects.empty() && keypoints.size() % rects.size() == 0)
		{
			keypointsPerBox = static_cast<int>(keypoints.size() / rects.size());
		}

		for (size_t i = 0; i < rects.size(); ++i)
		{
			const int category = hasCategoryPerBox ? categories[i] : -1;
			const cv::Scalar color = getColorForCategory(category);
			cv::Point2f pts[4];
			rects[i].points(pts);

			std::vector<cv::Point> contour;
			contour.reserve(4);
			for (const auto& pt : pts)
			{
				contour.emplace_back(cvRound(pt.x), cvRound(pt.y));
			}
			cv::polylines(vis, contour, true, color, 2, cv::LINE_AA);

			const cv::Point2f* anchor = &pts[0];
			for (int j = 1; j < 4; ++j)
			{
				if (pts[j].y < anchor->y || (pts[j].y == anchor->y && pts[j].x < anchor->x))
				{
					anchor = &pts[j];
				}
			}

			const int textY = (static_cast<int>(anchor->y) - 8 < 20) ? 20 : (static_cast<int>(anchor->y) - 8);
			cv::putText(
				vis,
				getCategoryDisplayName(category),
				cv::Point(static_cast<int>(anchor->x), textY),
				cv::FONT_HERSHEY_SIMPLEX,
				0.7,
				color,
				2,
				cv::LINE_AA);

			if (keypointsPerBox > 0)
			{
				const size_t startIndex = i * static_cast<size_t>(keypointsPerBox);
				const size_t endIndex = std::min(keypoints.size(), startIndex + static_cast<size_t>(keypointsPerBox));
				cv::Point prevPoint;
				bool hasPrevPoint = false;
				for (size_t kpIndex = startIndex; kpIndex < endIndex; ++kpIndex)
				{
					const cv::Point center(cvRound(keypoints[kpIndex].x), cvRound(keypoints[kpIndex].y));
					cv::circle(vis, center, 4, color, -1, cv::LINE_AA);
					if (hasPrevPoint)
					{
						cv::line(vis, prevPoint, center, color, 2, cv::LINE_AA);
					}
					prevPoint = center;
					hasPrevPoint = true;
				}
			}
		}

		return vis;
	}

	renderParam makeRenderParam(const std::vector<cv::RotatedRect>& rects, const std::vector<cv::Point2f>& dots);
	void beginProcessXml(CMarkup& xml, DeepLearning_AI_pose* deepAI, const std::string& apiName, const char* moduleStatusCh);
	void finishProcessXml(CMarkup& xml, char** xmlOut);
	void addScoreCategoryXml(CMarkup& xml, const std::string& score, const std::string& category, bool trimTrailingComma);
	void addAxisXml(CMarkup& xml, const std::vector<cv::RotatedRect>& rects, bool useRectAngle, const char* shortAxisName, const char* longAxisName);
	void addRenderXml(CMarkup& xml, renderParam& rendPar, const char* rectCh, bool includeDots);
	std::string trimTrailingCommaOrDefault(const std::string& value, const char* defaultValue);

	// 将扩展名统一转成小写，避免大小写路径导致的后端识别误差
	std::string toLowerCopy(const std::string& value)
	{
		std::string lower = value;
		std::transform(lower.begin(), lower.end(), lower.begin(),
			[](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
		return lower;
	}

	// ONNXRuntime 的张量类型先映射到 torch 类型，便于后续统一走 torch::Tensor 后处理链路
	torch::ScalarType ortTensorElementTypeToTorch(ONNXTensorElementDataType element_type)
	{
		switch (element_type)
		{
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			return torch::kFloat32;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			return torch::kFloat64;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			return torch::kInt64;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			return torch::kInt32;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			return torch::kUInt8;
		default:
			throw std::runtime_error("Unsupported ONNX tensor element type");
		}
	}

	// ONNXRuntime 输出张量默认由 ORT 管理生命周期，这里 clone 一份后续就能安全脱离会话对象使用
	torch::Tensor ortValueToTensor(Ort::Value& value)
	{
		auto type_info = value.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> shape = type_info.GetShape();
		const auto element_type = type_info.GetElementType();
		const torch::ScalarType torch_type = ortTensorElementTypeToTorch(element_type);

		if (shape.empty())
		{
			shape.push_back(1);
		}

		switch (element_type)
		{
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			return torch::from_blob(value.GetTensorMutableData<float>(), shape, torch_type).clone();
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			return torch::from_blob(value.GetTensorMutableData<double>(), shape, torch_type).clone();
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			return torch::from_blob(value.GetTensorMutableData<int64_t>(), shape, torch_type).clone();
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			return torch::from_blob(value.GetTensorMutableData<int32_t>(), shape, torch_type).clone();
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			return torch::from_blob(value.GetTensorMutableData<uint8_t>(), shape, torch_type).clone();
		default:
			throw std::runtime_error("Unsupported ONNX tensor element type");
		}
	}

	// 从 XML 中读取 moduleID，并在全局实例表里找到对应的模型对象
	DeepLearning_AI_pose* findDeepAIByModuleXml(char* xmlIn, const std::string& apiName)
	{
		ResolveXml rXml;
		int id = rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");
		auto mapIter = g_mapDLE_AI_pose.find(id);
		if (mapIter == g_mapDLE_AI_pose.end())
			return nullptr;

		return mapIter->second;
	}

	// 解析处理阶段 XML，并把 rotRectROI 统一折算成普通矩形 ROI，供后续裁剪与回填坐标使用
	void prepareProcessXmlInput(DeepLearning_AI_pose* deepAI, void* inImg, char* xmlIn, const std::string& apiName)
	{
		int ret = deepAI->parseProcessXml(inImg, xmlIn, apiName);
		if (ret)
			throw "parseProcessXml Error";

		deepAI->rotateRect2Rect();
	}

	// 调试模式下保存源图与 ROI，方便定位 XML 传入范围是否正确
	void saveDebugSourceImage(DeepLearning_AI_pose* deepAI)
	{
		if (!deepAI->m_pDebugProcess)
			return;

		cv::Mat buf = deepAI->m_pSrcImg.clone();
		cv::rectangle(buf, deepAI->m_pRect, cv::Scalar(255, 0, 0), 2, 8);
		cv::imwrite("d:\\debug\\spring_Src.jpg", buf);
	}

	// 调试模式下保存结果图，方便定位 XML 传入范围是否正确
	void saveDebugResultImage(
		DeepLearning_AI_pose* deepAI,
		const std::vector<cv::RotatedRect>& rects,
		const std::vector<cv::Point2f>& dots)
	{
		if (!deepAI->m_pDebugResult)
			return;

		cv::Mat buf = deepAI->m_pSrcImg.clone();

		cv::rectangle(buf, deepAI->m_pRect, cv::Scalar(255, 0, 0), 2, 8);

		for (const auto& rect : rects)
		{
			cv::Point2f pts[4];
			rect.points(pts);
			for (int i = 0; i < 4; ++i)
			{
				cv::line(buf, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 0, 255), 2, 8);
			}
		}

		for (const auto& pt : dots)
		{
			cv::circle(buf, pt, 3, cv::Scalar(0, 255, 0), -1, 8);
		}

		cv::imwrite("d:\\debug\\spring_Result.jpg", buf);
	}

	// 若调用方传入 rotRectROI，则截取首个 ROI 对应子图；否则直接使用整图
	cv::Mat cropByPreparedRectRoi(DeepLearning_AI_pose* deepAI, cv::Rect& rectSrc)
	{
		// prepareProcessXmlInput 已经通过 rotateRect2Rect() 把首个旋转 ROI 折算到 m_pRect
		rectSrc = deepAI->m_pRect;
		return deepAI->m_pSrcImg(rectSrc);
	}


	// 统一生成“检测到目标/缺陷”的结果 XML
	void writeObjectResultXml(
		char** xmlOut,
		DeepLearning_AI_pose* deepAI,
		const std::string& apiName,
		int defectStatus,
		const std::vector<cv::RotatedRect>& rects,
		const std::vector<cv::Point2f>& dots,
		const std::string& score,
		const std::string& category,
		bool trimScoreCategory,
		bool useRectAngle,
		const char* shortAxisName,
		const char* longAxisName,
		const char* moduleStatusCh,
		const char* rectCh)
	{
		CMarkup xml;
		renderParam rendPar = makeRenderParam(rects, dots);

		beginProcessXml(xml, deepAI, apiName, moduleStatusCh);
		xml.AddElem("defectStatus", defectStatus);
		xml.SetAttrib("CH", "结果：存在缺陷");
		addRenderXml(xml, rendPar, rectCh, true);
		addScoreCategoryXml(xml, score, category, trimScoreCategory);
		addAxisXml(xml, rects, useRectAngle, shortAxisName, longAxisName);
		finishProcessXml(xml, xmlOut);
	}

	// 统一生成“未检测到目标/结果正常”的 XML
	void writeNormalResultXml(
		char** xmlOut,
		DeepLearning_AI_pose* deepAI,
		const std::string& apiName,
		int defectStatus,
		bool includeRender,
		bool includeAxis,
		const char* shortAxisName,
		const char* longAxisName)
	{
		CMarkup xml;
		beginProcessXml(xml, deepAI, apiName, "模块状态");
		xml.AddElem("defectStatus", defectStatus);
		xml.SetAttrib("CH", "结果：正常");

		if (includeRender)
		{
			xml.AddElem("renderRectBox", "");
			xml.SetAttrib("CH", "渲染矩形");
			xml.AddElem("renderDot", "");
			xml.SetAttrib("CH", "渲染点集");
		}

		xml.AddElem("score", 0);
		xml.SetAttrib("CH", "置信度分数");
		xml.AddElem("category", -100);
		xml.SetAttrib("CH", "类别");

		if (includeAxis)
		{
			xml.AddElem(shortAxisName, "");
			xml.SetAttrib("CH", "短轴");
			xml.AddElem(longAxisName, "");
			xml.SetAttrib("CH", "长轴");
			xml.AddElem("angle", "");
			xml.SetAttrib("CH", "角度");
		}

		finishProcessXml(xml, xmlOut);
	}

	// 输出统一的整条调用耗时，便于从 DLL 日志观察端到端性能
	void logWholeProcessCost(DWORD startTick)
	{
		DWORD endTick = GetTickCount64();
		string str = "The whole process Running Time is :";
		str.append(to_string(endTick - startTick));
		DeepLearning_AI_pose::addAiLog(LOG_MESSAGE, str.c_str());
	}

	// 将矩形框与点集打包成 renderParam，统一供 XML 输出函数复用
	renderParam makeRenderParam(const std::vector<cv::RotatedRect>& rects, const std::vector<cv::Point2f>& dots)
	{
		renderParam rendPar;
		rendPar.m_renderRectBox = rects;
		rendPar.m_renderDot = dots;
		return rendPar;
	}

	// 创建 XML 根结构并写入公共的模块状态字段
	void beginProcessXml(CMarkup& xml, DeepLearning_AI_pose* deepAI, const std::string& apiName, const char* moduleStatusCh)
	{
		xml.SetDoc("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n");
		xml.AddElem("Param");
		xml.IntoElem();
		xml.AddElem(apiName.c_str());
		xml.IntoElem();

		CString statusNumStr;
		int modStatus = deepAI->m_pModStatus ? 1 : 0;
		statusNumStr.Format("%d", modStatus);
		xml.AddElem("moduleStatus", statusNumStr.GetBuffer());
		xml.SetAttrib("CH", moduleStatusCh);
	}

	// 结束 XML 节点并把完整字符串写回调用方提供的缓冲区
	void finishProcessXml(CMarkup& xml, char** xmlOut)
	{
		xml.OutOfElem();
		CString Out = xml.GetDoc();
		strcpy(*xmlOut, Out.GetBuffer());
	}

	// 统一写入 score/category 字段，兼容“是否需要裁掉末尾逗号”两种来源
	void addScoreCategoryXml(CMarkup& xml, const std::string& score, const std::string& category, bool trimTrailingComma)
	{
		const std::string scoreText = trimTrailingComma ? trimTrailingCommaOrDefault(score, "0") : score;
		const std::string categoryText = trimTrailingComma ? trimTrailingCommaOrDefault(category, "-100") : category;

		xml.AddElem("score", scoreText.c_str());
		xml.SetAttrib("CH", "置信度分数");
		xml.AddElem("category", categoryText.c_str());
		xml.SetAttrib("CH", "类别");
	}

	// 由检测框批量计算短轴、长轴和角度字符串，便于多个 XML 输出入口共用
	void buildAxisStrings(
		const std::vector<cv::RotatedRect>& rects,
		bool useRectAngle,
		std::string& shortAxisStr,
		std::string& longAxisStr,
		std::string& angleStr)
	{
		shortAxisStr.clear();
		longAxisStr.clear();
		angleStr.clear();

		for (const auto& rect : rects)
		{
			const float shortAxis = std::min(rect.size.width, rect.size.height);
			const float longAxis = std::max(rect.size.width, rect.size.height);
			const float angle = useRectAngle ? rect.angle : 0.0f;

			shortAxisStr.append(to_string(shortAxis)).append(",");
			longAxisStr.append(to_string(longAxis)).append(",");
			angleStr.append(to_string(angle)).append(",");
		}

		if (!shortAxisStr.empty())
		{
			shortAxisStr.pop_back();
			longAxisStr.pop_back();
			angleStr.pop_back();
		}
	}

	// 将短轴、长轴和角度三个字段一次性写入 XML
	void addAxisXml(
		CMarkup& xml,
		const std::vector<cv::RotatedRect>& rects,
		bool useRectAngle,
		const char* shortAxisName,
		const char* longAxisName)
	{
		std::string shortAxisStr;
		std::string longAxisStr;
		std::string angleStr;
		buildAxisStrings(rects, useRectAngle, shortAxisStr, longAxisStr, angleStr);

		xml.AddElem(shortAxisName, shortAxisStr.c_str());
		xml.SetAttrib("CH", "短轴");
		xml.AddElem(longAxisName, longAxisStr.c_str());
		xml.SetAttrib("CH", "长轴");
		xml.AddElem("angle", angleStr.c_str());
		xml.SetAttrib("CH", "角度");
	}

	// 将 renderRectBox / renderDot 这类渲染字段统一写入 XML
	void addRenderXml(CMarkup& xml, renderParam& rendPar, const char* rectCh, bool includeDots)
	{
		xml.AddElem("renderRectBox", rendPar.conRenderRect().c_str());
		xml.SetAttrib("CH", rectCh);

		if (includeDots)
		{
			xml.AddElem("renderDot", rendPar.conRenderDot().c_str());
			xml.SetAttrib("CH", "渲染点集");
		}
	}

	// 某些历史字段以逗号拼接保存；这里负责在非空时去掉末尾逗号，否则返回默认值
	std::string trimTrailingCommaOrDefault(const std::string& value, const char* defaultValue)
	{
		if (value.empty())
			return defaultValue;
		return value.substr(0, value.length() - 1);
	}
}

// 通过扩展名判断是否为 ONNX 模型文件
bool DeepLearning_AI_pose::isOnnxModelPath(const std::string& module_path)
{
	const std::string extension = toLowerCopy(std::filesystem::path(module_path).extension().string());
	return extension == ".onnx";
}

// 通过扩展名判断是否为 TorchScript 模型文件
bool DeepLearning_AI_pose::isTorchScriptModelPath(const std::string& module_path)
{
	const std::string extension = toLowerCopy(std::filesystem::path(module_path).extension().string());
	return extension == ".torchscript";
}

// 根据模型文件扩展名选择具体推理后端
bool DeepLearning_AI_pose::inferModelRuntimeFormat(const std::string& module_path, ModelRuntimeFormat& runtime_format)
{
	if (isOnnxModelPath(module_path))
	{
		runtime_format = ModelRuntimeFormat::Onnx;
		return true;
	}

	if (isTorchScriptModelPath(module_path))
	{
		runtime_format = ModelRuntimeFormat::TorchScript;
		return true;
	}

	return false;
}

// 切换模型前先清空状态，避免上一个模型残留的 TorchScript/ONNX 资源污染当前实例
void DeepLearning_AI_pose::resetRuntimeState()
{
	m_pIsLoadedModule = false;
	m_pModStatus = false;
	m_pModule = torch::jit::Module();
	m_modelRuntimeFormat = ModelRuntimeFormat::TorchScript;
	resetOnnxRuntimeState();
}

// 初始化 ONNXRuntime 会话，并缓存输入/输出名字，后续推理时不再重复查询元数据
int DeepLearning_AI_pose::loadOnnxModule()
{
	m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DeepLearning_AI_pose");
	const std::filesystem::path model_path = std::filesystem::path(m_pModulePath);
	const auto buildSessionOptions = []() {
		Ort::SessionOptions options;
		options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		options.SetIntraOpNumThreads(2);
		return options;
	};

	m_ortUsingCuda = false;
	m_ortCudaDeviceId = 0;

	Ort::SessionOptions session_options = buildSessionOptions();
	bool requestedCuda = false;

	if (m_pDeviceType)
	{
		requestedCuda = true;
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = static_cast<int>(gpuID);

		try
		{
			//告诉 ONNX Runtime 这个模型会话优先使用 NVIDIA GPU/CUDA 来跑推理
			//能用 GPU 的算子放到 GPU 上跑，不能用的算子可能回退到 CPU
			session_options.AppendExecutionProvider_CUDA(cuda_options);
			m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, model_path.c_str(), session_options);
			m_ortUsingCuda = true;
			m_ortCudaDeviceId = cuda_options.device_id;
		}
		catch (const std::exception& e)
		{
			std::cout
				<< "ONNXRuntime CUDAExecutionProvider unavailable for cuda:" << static_cast<int>(gpuID)
				<< ", fallback to CPUExecutionProvider. reason: " << e.what() << std::endl;
			m_ortSession.reset();
			session_options = buildSessionOptions();
		}
	}

	if (!m_ortSession)
	{
		m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, model_path.c_str(), session_options);
		if (requestedCuda)
		{
			std::cout << "ONNXRuntime session is running on CPUExecutionProvider." << std::endl;
		}
	}

	Ort::AllocatorWithDefaultOptions allocator;
	m_ortInputNames.clear();
	m_ortOutputNames.clear();

	const size_t input_count = m_ortSession->GetInputCount();
	for (size_t index = 0; index < input_count; ++index)
	{
		auto name = m_ortSession->GetInputNameAllocated(index, allocator);
		m_ortInputNames.emplace_back(name.get());
	}

	const size_t output_count = m_ortSession->GetOutputCount();
	for (size_t index = 0; index < output_count; ++index)
	{
		auto name = m_ortSession->GetOutputNameAllocated(index, allocator);
		m_ortOutputNames.emplace_back(name.get());
	}

	if (m_ortInputNames.empty() || m_ortOutputNames.empty())
	{
		throw std::runtime_error("ONNX model has no input or output nodes.");
	}

	const int fixedInputSize = getFixedSquareOnnxInputSize(*m_ortSession);
	if (fixedInputSize > 0 && m_inputSize > 0 && fixedInputSize != m_inputSize)
	{
		throw std::runtime_error(buildOnnxInputSizeHint(m_inputSize, fixedInputSize));
	}

	m_modelRuntimeFormat = ModelRuntimeFormat::Onnx;
	return 0;
}

// ONNX 相关缓存统一在这里清理，方便 resetRuntimeState 直接复用
void DeepLearning_AI_pose::resetOnnxRuntimeState()
{
	m_ortSession.reset();
	m_ortEnv.reset();
	m_ortInputNames.clear();
	m_ortOutputNames.clear();
	m_ortUsingCuda = false;
	m_ortCudaDeviceId = 0;
}

// ============================================================================
// 检测模型初始化入口
// 流程：
//   1. 从 XML 解析 moduleID
//   2. 若全局 map 中不存在该 ID，则创建新的 CDeepAI 实例并插入 map
//   3. 调用 loadModule 加载模型文件
//   4. 将初始化状态输出为 XML
// 返回值：0=成功，非 0=失败
// ============================================================================
int DeepLearning_AI_pose::initPoseAiObj(char* xmlIn, char** xmlOut, string apiName)
{
	int ret = 0;
	int id = 0;
	std::string modulePath;

	DeepLearning_AI_pose* deepAI = nullptr;

	id = parseInitPoseXml(xmlIn, apiName);

	// 如果全局映射表中不存在该 ID，则创建新对象并插入映射表
	auto mapIter = g_mapDLE_AI_pose.find(id);
	if (mapIter != g_mapDLE_AI_pose.end())
		deepAI = mapIter->second;
	else
	{
		deepAI = new DeepLearning_AI_pose;
		g_mapDLE_AI_pose.insert(std::pair<int, DeepLearning_AI_pose*>(id, deepAI));
	}

	//deepAI->m_pMutex.lock();
	ret = deepAI->loadModule(xmlIn, apiName);

	deepAI->outputInitXml(*xmlOut, apiName);
	//deepAI->m_pMutex.unlock();

	return ret;
}

// 从初始化 XML 中解析出 moduleID
int DeepLearning_AI_pose::parseInitPoseXml(char* xmlIn, std::string apiName)
{
	int ret = 0;
	ResolveXml rXml;

	return rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");
}

// 生成模型初始化结果 XML，只包含模块状态字段
int DeepLearning_AI_pose::outputInitXml(char* xmlOut, string apiName)
{
	// 构建 XML 字符串
	CMarkup xml;
	xml.SetDoc("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n");
	xml.AddElem("Param");
	xml.IntoElem();

	xml.AddElem(apiName.c_str());
	xml.IntoElem();
	CString initNumStr;

	// 模块状态
	int modStatus = m_pModStatus ? 1 : 0;
	initNumStr.Format("%d", modStatus);
	xml.AddElem("moduleStatus", initNumStr.GetBuffer());
	xml.SetAttrib("CH", "模块状态");

	xml.OutOfElem();
	CString Out = xml.GetDoc();
	strcpy(xmlOut, Out.GetBuffer());
	return 0;
}

// ============================================================================
// 销毁指定 moduleID 对应的模型实例
// 流程：
//   1. 解析 XML 获取 moduleID
//   2. 在全局 map 中查找并删除 DeepLearning_AI_pose 实例
//   3. 调用 outputDesXml 输出销毁状态 XML
// 注意：
// 返回值：0=成功，-4=模型未找到
// ============================================================================
int DeepLearning_AI_pose::destoryPoseObject(char* xmlIn, char** xmlOut, std::string apiName)
{
	DeepLearning_AI_pose* deepAI = nullptr;
	ResolveXml rXml;
	int id = rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");

	// 在全局 map 中查找目标模型实例
	auto mapIter = g_mapDLE_AI_pose.find(id);
	if (mapIter == g_mapDLE_AI_pose.end())
		return -4;

	deepAI = mapIter->second;
	//销毁后返回的 XML 里模块状态为 0
	deepAI->m_pModStatus = false;
	// 释放模型对象内存
	// 注意：此处未从 g_mapDLE_AI_pose 中 erase 该条目，后续访问会产生悬空指针；
	// 实际使用时需确保 destory 后不再引用该 moduleID
	deepAI->outputDesXml(*xmlOut, apiName);
	delete deepAI;
	g_mapDLE_AI_pose.erase(mapIter);

	return 0;
}

// 输出销毁操作的 XML 状态报告
void DeepLearning_AI_pose::outputDesXml(char* xmlOut, std::string apiName)
{
	// 构建 XML 字符串
	CMarkup xml;
	xml.SetDoc("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n");
	xml.AddElem("Param");
	xml.IntoElem();

	xml.AddElem(apiName.c_str());
	xml.IntoElem();
	CString initNumStr;

	// 模块状态
	int desStatu = m_pModStatus ? 1 : 0;
	initNumStr.Format("%d", desStatu);
	xml.AddElem("moduleStatus", initNumStr.GetBuffer());
	xml.SetAttrib("CH", "模块状态");

	xml.OutOfElem();
	CString Out = xml.GetDoc();
	strcpy(xmlOut, Out.GetBuffer());
}

// ============================================================================
// 解析初始化配置、加载模型后端，并完成预热
// 流程：
//   1. 解析 XML，读取 deviceType、modulePath、gpuID、YOLO 版本和输入尺寸
//   2. 若模型路径未变化且当前实例已处于可用状态，则直接复用已加载模型
//   3. 重置上一次加载留下的运行时状态，并记录新的模型路径
//   4. 若 inputSize 未显式指定(==0)，则按任务类型补默认输入尺寸
//   5. 根据模型文件扩展名选择 ONNXRuntime 或 TorchScript 后端，并执行实际加载
//   6. 按当前任务构造预热输入张量
//   7. 执行 warmup，并输出当前推理后端信息
// ============================================================================
int DeepLearning_AI_pose::loadModule(char* xmlIn, string apiName)
{
	try
	{
		int ret = 0;
		string modulePath = "";
		ResolveXml rXml;

		m_pDeviceType = rXml.parseXmlInt(xmlIn, "Param", apiName, "deviceType");
		modulePath = rXml.parseXmlStr(xmlIn, "Param", apiName, "modulePath");
		gpuID = rXml.parseXmlInt(xmlIn, "Param", apiName, "gpuID");
		if (gpuID < 0)
		{
			gpuID = 0;
		}
		// YOLO 版本选择：0 表示 YOLOv8，1 表示 YOLOv5
		//m_yolo_version = rXml.parseXmlInt(xmlIn, "Param", apiName, "m_yolo_version");
		m_inputSize = rXml.parseXmlInt(xmlIn, "Param", apiName, "inputSize");
		// 如果 xmlIn 中的路径与已保存路径一致
		// 相同路径且当前实例已处于可用状态时，直接复用，避免重复加载模型
		if (!strcmp(modulePath.c_str(), m_pModulePath.c_str()))
		{
			// 如果模型已加载，则直接返回
			if (m_pIsLoadedModule && m_pModStatus)
				return 0;
		}
		resetRuntimeState();
		m_pModulePath = modulePath;

		torch::set_num_threads(2);

		// 没有显式传 inputSize 时，补Pose模型的默认输入尺寸
		if (m_inputSize == 0) {
			m_inputSize = 640; // 姿态Pose
		}

		ModelRuntimeFormat runtime_format = ModelRuntimeFormat::TorchScript;
		if (!inferModelRuntimeFormat(m_pModulePath, runtime_format))
		{
			throw std::runtime_error(
				"Unsupported model file extension. Expected .onnx, .torchscript, or .pt.");
		}

		// 后端选择只在加载时决定一次，后续推理统一走 forwardWithTiming
		if (runtime_format == ModelRuntimeFormat::Onnx)
		{
			loadOnnxModule();
		}
		else
		{
			if (m_pDeviceType && torch::hasCUDA())
			{
				m_pDevice = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(gpuID));
			}
			else
			{
				m_pDevice = c10::Device(c10::kCPU);
			}
			m_pModule = torch::jit::load(m_pModulePath.c_str(), m_pDevice);
			m_modelRuntimeFormat = ModelRuntimeFormat::TorchScript;
			if (m_pDeviceType && m_pDevice.type() != c10::kCUDA)
			{
				std::cout
					<< "TorchScript CUDAExecutionProvider unavailable for cuda:" << static_cast<int>(gpuID)
					<< ", fallback to CPUExecutionProvider." << std::endl;
			}
		}

		std::vector<int64_t> warmupShape = { 1, 3, m_inputSize, m_inputSize };
		auto img = torch::rand(warmupShape).to(m_pDevice);
		std::cout << "img size: " << img.sizes() << std::endl;
		DeepLearning_AI_pose::warmup(img);

		std::string backendMessage;
		if (m_modelRuntimeFormat == ModelRuntimeFormat::Onnx)
		{
			if (m_ortUsingCuda)
			{
				backendMessage = "Current inference backend: ONNXRuntime / GPU (cuda:";
				backendMessage.append(std::to_string(m_ortCudaDeviceId));
				backendMessage.append(")");
			}
			else
			{
				backendMessage = "Current inference backend: ONNXRuntime / CPU";
				if (m_pDeviceType)
				{
					backendMessage.append(" (requested GPU, actual provider: CPUExecutionProvider)");
				}
			}
		}
		else
		{
			const bool useCuda = (m_pDevice.type() == c10::kCUDA);
			backendMessage = useCuda
				? "Current inference backend: TorchScript / GPU"
				: "Current inference backend: TorchScript / CPU";
			if (useCuda)
			{
				backendMessage.append(" (cuda:");
				backendMessage.append(std::to_string(static_cast<int>(m_pDevice.index())));
				backendMessage.append(")");
			}
			else if (m_pDeviceType)
			{
				backendMessage.append(" (requested GPU, actual provider: CPUExecutionProvider)");
			}
		}
		std::cout << backendMessage << std::endl;

	}
	catch (const std::exception& e)
	{
		std::string errorMessage = e.what();
		errorMessage.append(buildInputSizeMismatchHint(errorMessage, m_inputSize));
		std::cerr << "loadModule std Exception: " << errorMessage << std::endl;
		CString str;
		str.Format("loadModule std Exception %s", errorMessage.c_str());
		m_pIsLoadedModule = false;
		m_pModStatus = false;
		return -1;
	}

	m_pIsLoadedModule = true;
	m_pModStatus = true;
	return 0;
}

// warmup — GPU/CUDA 预热推理
// 执行两次空 forward 以触发 CUDA kernel 编译和显存分配，
// 避免正式推理时因延迟初始化导致耗时波动
// 参数 img：与模型输入尺寸匹配的随机张量
void DeepLearning_AI_pose::warmup(at::Tensor img)
{
	auto start = std::chrono::high_resolution_clock::now();

	if (m_modelRuntimeFormat == ModelRuntimeFormat::Onnx)
	{
		warmupOnnx(img);
	}
	else
	{
		// 推理
		torch::jit::IValue output = m_pModule.forward({ img });
		torch::jit::IValue output1 = m_pModule.forward({ img });
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "warmup cost : " << duration.count() << " ms" << std::endl;

}

// ONNX 预热与 TorchScript 保持一致：执行两次空推理，尽量把初始化成本前置
void DeepLearning_AI_pose::warmupOnnx(const torch::Tensor& img)
{
	(void)forwardOnnx(img);
	(void)forwardOnnx(img);
}

// 统一把输入拉回 CPU 连续内存再喂给 ONNXRuntime，避免不同设备/步长下的数据兼容问题
std::vector<torch::Tensor> DeepLearning_AI_pose::forwardOnnx(const torch::Tensor& tensor_img)
{
	if (!m_ortSession)
	{
		throw std::runtime_error("ONNX session is not initialized.");
	}

	torch::Tensor cpu_tensor = tensor_img.detach().to(torch::kCPU).contiguous();
	std::vector<int64_t> input_shape;
	for (const auto dim : cpu_tensor.sizes())
	{
		input_shape.push_back(dim);
	}

	// 这里直接引用 cpu_tensor 的内存，不再做额外复制
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		cpu_tensor.data_ptr<float>(),
		static_cast<size_t>(cpu_tensor.numel()),
		input_shape.data(),
		input_shape.size());

	std::vector<const char*> input_names;
	for (const auto& name : m_ortInputNames)
	{
		input_names.push_back(name.c_str());
	}

	std::vector<const char*> output_names;
	for (const auto& name : m_ortOutputNames)
	{
		output_names.push_back(name.c_str());
	}

	auto ort_outputs = m_ortSession->Run(
		Ort::RunOptions{ nullptr },
		input_names.data(),
		&input_tensor,
		1,
		output_names.data(),
		output_names.size());

	std::vector<torch::Tensor> outputs;
	outputs.reserve(ort_outputs.size());
	for (auto& output : ort_outputs)
	{
		outputs.push_back(ortValueToTensor(output));
	}
	return outputs;
}

// 姿态关键点检测Pose任务处理主流程
int DeepLearning_AI_pose::processOneImgByPose(void* inImg, void*& outImg, char* xmlIn, char** xmlOut, std::string apiName)
{
	DeepLearning_AI_pose* deepAI = findDeepAIByModuleXml(xmlIn, apiName);
	if (deepAI == nullptr)
		return -4;
	try
	{
		DWORD t1 = GetTickCount64();
		prepareProcessXmlInput(deepAI, inImg, xmlIn, apiName);
		saveDebugSourceImage(deepAI);

		cv::Rect rectSrc;
		cv::Mat cjImg = cropByPreparedRectRoi(deepAI, rectSrc);

		std::vector<cv::RotatedRect> DC_VecRotaRect;
		std::string DC_score, DC_class;
		std::vector<cv::Point2f> DC_Pts;
		int result = 0;

		result = deepAI->runPose(cjImg, DC_VecRotaRect, DC_score, DC_class, DC_Pts, rectSrc);

		//deepAI->m_pDrcImg = renderPoseImage(
		//	deepAI->m_pSrcImg,
		//	DC_VecRotaRect,
		//	parseCategoryIds(DC_class),
		//	DC_Pts);
		// 
		//// 输出图像构造
		//m_renderparam.conOutImg(deepAI->m_pDrcImg, outImg);

		saveDebugResultImage(deepAI, DC_VecRotaRect, DC_Pts);

		int DC_run = 1;
		if (result > 0)
		{
			DC_run = 2;
			writeObjectResultXml(
				xmlOut,
				deepAI,
				apiName,
				DC_run,
				DC_VecRotaRect,
				DC_Pts,
				DC_score,
				DC_class,
				true,
				false,
				"shortAxis",
				"longAxis",
				"模块状态",
				"渲染矩形");
		}
		else
		{
			writeNormalResultXml(xmlOut, deepAI, apiName, DC_run, true, false, "shortAxis", "longAxis");
		}

		logWholeProcessCost(t1);
	}
	catch (const std::exception& e)
	{
		CString str;
		str.Format("processOneImg std Exception %s", e.what());
		DeepLearning_AI_pose::addAiLog(LOG_ERROR, str.GetBuffer());
		return -2;
	}
	catch (cv::Exception& e)
	{
		CString str;
		str.Format("processOneImg cv Exception %s", e.what());
		DeepLearning_AI_pose::addAiLog(LOG_ERROR, str.GetBuffer());
		return -3;
	}
	return 0;
}

int DeepLearning_AI_pose::outputPoseImg(cv::Mat inImg, void*& outImg)
{
	int size = inImg.total() * inImg.elemSize();
	unsigned char* dstMat = new unsigned char[size];
	std::memcpy(dstMat, inImg.data, size * sizeof(unsigned char));
	outImg = new cv::Mat(inImg.rows, inImg.cols, inImg.type(), dstMat);
	return 0;
}

// ============================================================================
// 姿态估计(pose)模型推理与后处理
// 输出 YOLO 姿态格式：每个检测包含 bbox + num_kps 个关键点坐标
// 后处理流程：
//   1. squeeze + transpose → 过滤低置信度预测
//   2. (cx,cy,w,h) → (x1,y1,x2,y2) 格式转换
//   3. 提取 bbox / score / class / pts
//   4. 按 类别为 0 过滤 → NMS 去重
//   5. 去 padding 并还原到原图坐标 → 填充 DC_VecRotaRect / DC_score / DC_class / DC_Pts
// 返回值：检测到的目标数量（0 表示无结果）
// ============================================================================
int DeepLearning_AI_pose::runPose(cv::Mat& m_pSrcImg, std::vector<cv::RotatedRect>& DC_VecRotaRect, std::string& DC_score, std::string& DC_class, std::vector<cv::Point2f>& DC_Pts, cv::Rect& rectSrc) {

	m_pPoseKptScore.clear();
	m_pPoseNumKpts = 0;
	float conf_thres = m_pThreshold; float iou_thres = 0.7f;
	if (conf_thres > 1.0) conf_thres = 0.8;
	int image_size = m_inputSize;
	torch::NoGradGuard no_grad;
	// TODO：检查图像尺寸

	/*** 预处理 ***/
	InferenceInput input = prepareYoloInput(m_pSrcImg, image_size);

	const float pad_w = input.pad_w;

	const float pad_h = input.pad_h;
	const float scale = input.scale;

	auto outputs = forwardWithTiming(input.tensor);
	if (outputs.empty())
	{
		return 0;
	}
	auto output = outputs[0];

	/*** 后处理 ***/
	auto post_start = std::chrono::high_resolution_clock::now();

	torch::Tensor pred = output.squeeze().transpose(0, 1);
	PoseLayout layout;
	const int feature_dim = static_cast<int>(pred.size(1));
	if (!inferPoseLayout(feature_dim, layout)) {
		std::cerr << "runPose unsupported feature dimension: " << feature_dim << std::endl;
		return 0;
	}
	m_pPoseNumKpts = layout.num_kps;

	torch::Tensor det = buildPoseDetections(pred, conf_thres, iou_thres, layout);
	int num_det = det.size(0);
	if (num_det == 0) return 0;

	appendPoseDetections(det, layout, pad_w, pad_h, scale, rectSrc, DC_VecRotaRect, DC_score, DC_class, DC_Pts);

	logStageCost("post-process", post_start);

	return num_det;
}

// 根据 feature_dim 反推姿态输出布局
// 这里兼容 [x, y, score] 和 [x, y] 两种关键点表示
bool DeepLearning_AI_pose::inferPoseLayout(int feature_dim, PoseLayout& layout)
{
	layout = { 0, 0, 0 };

	for (int candidate_nc : { 2, 1 }) {
		const int remain = feature_dim - 4 - candidate_nc;
		if (remain > 0 && remain % 3 == 0) {
			layout = { candidate_nc, remain / 3, 3 };
			return true;
		}
	}

	for (int candidate_nc : { 2, 1 }) {
		const int remain = feature_dim - 4 - candidate_nc;
		if (remain > 0 && remain % 2 == 0) {
			layout = { candidate_nc, remain / 2, 2 };
			return true;
		}
	}

	return false;
}

// 姿态后处理：
// 先按类别置信度筛选，再做 NMS，最后把关键点跟随保留下来
torch::Tensor DeepLearning_AI_pose::buildPoseDetections(torch::Tensor pred, float conf_thres, float iou_thres, const PoseLayout& layout)
{
	std::vector<torch::Tensor> indices = torch::where(pred.slice(1, 4, 4 + layout.nc) > conf_thres);
	auto index_i = indices[0], index_j = indices[1];
	if (index_i.size(0) == 0) {
		return torch::empty({ 0, 0 });
	}

	pred = pred.index({ index_i });
	xywhToXyxyInPlace(pred);

	auto boxes = pred.slice(1, 0, 4);
	auto scores = pred.gather(1, (index_j + 4).unsqueeze(1));
	auto class_idxs = index_j.to(torch::kFloat32).unsqueeze(1);
	auto pts = pred.slice(1, 4 + layout.nc);

	auto det = torch::cat({ boxes, scores, class_idxs, pts }, 1).cpu();
	constexpr float max_wh = 8192.0f;
	auto boxes_for_nms = det.slice(1, 0, 4) + det.select(1, 5).unsqueeze(1) * max_wh;
	auto keep = nms_kernel(boxes_for_nms, det.select(1, 4), iou_thres);
	return torch::index_select(det, 0, keep);
}

// 将首个 rotRectROI 转成普通矩形
// 若未配置 ROI，则默认整图参与推理
void DeepLearning_AI_pose::rotateRect2Rect()
{
	bool emptyFlag = false;
	cv::RotatedRect rotateRectTem;

	if (m_pVecRotaRect.empty())
		emptyFlag = true;

	if (!emptyFlag)
		rotateRectTem = m_pVecRotaRect[0];

	if (rotateRectTem.center.x == 0 && rotateRectTem.center.y == 0 \
		&& rotateRectTem.size.width == 0 && rotateRectTem.size.height == 0 || emptyFlag)
	{
		m_pRect.x = 0;
		m_pRect.y = 0;
		m_pRect.width = m_pSrcImg.cols;
		m_pRect.height = m_pSrcImg.rows;
	}
	else
	{
		m_pRect.x = rotateRectTem.center.x - rotateRectTem.size.width / 2;
		m_pRect.y = rotateRectTem.center.y - rotateRectTem.size.height / 2;
		m_pRect.width = rotateRectTem.size.width;
		m_pRect.height = rotateRectTem.size.height;
	}
}

// 统一处理 YOLO 路径的 Letterbox 与 BCHW 张量转换
DeepLearning_AI_pose::InferenceInput DeepLearning_AI_pose::prepareYoloInput(cv::Mat& image, int image_size)
{
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<float> pad_info = LetterboxImage(image, image, cv::Size(image_size, image_size));
	auto tensor_img = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() }, torch::kUInt8).to(m_pDevice);
	tensor_img = tensor_img.flip(3).permute({ 0, 3, 1, 2 }).div(255).contiguous();

	logStageCost("pre-process", start);
	return { tensor_img, pad_info[0], pad_info[1], pad_info[2] };
}

// 输出单阶段耗时，阶段名由调用方传入
void DeepLearning_AI_pose::logStageCost(const char* stage_name, std::chrono::high_resolution_clock::time_point start)
{
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << stage_name << " takes : " << duration.count() * 1e-3 << " ms" << std::endl;
}

// 原地将中心点框表达转成左上/右下框表达，避免产生额外张量副本
void DeepLearning_AI_pose::xywhToXyxyInPlace(torch::Tensor& pred)
{
	pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2).div(2);
	pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3).div(2);
	pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
	pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);
}

// 通用 letterbox 预处理：
// 保持长宽比缩放后居中补边，同时返回 pad_w/pad_h/scale 供坐标反算使用
// ============================================================================
// LetterboxImage — 等比例缩放并居中填充图像到目标尺寸
//
// 保持宽高比的缩放 + 对称填充（灰色 114），返回左侧/上侧填充量和缩放比例
// 这是 YOLO 系列模型标准预处理步骤
//
// 参数 src：输入图像（只读）
// 参数 dst：输出图像（会被原地修改）
// 参数 out_size：目标尺寸（通常为正方形，如 640×640）
// 返回值：{pad_left, pad_top, scale} 三个浮点数
// ============================================================================
std::vector<float> DeepLearning_AI_pose::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
	auto in_h = static_cast<float>(src.rows);
	auto in_w = static_cast<float>(src.cols);
	float out_h = out_size.height;
	float out_w = out_size.width;

	float scale = std::min(out_w / in_w, out_h / in_h);

	int mid_h = static_cast<int>(in_h * scale);
	int mid_w = static_cast<int>(in_w * scale);

	cv::resize(src, dst, cv::Size(mid_w, mid_h));

	int top = (static_cast<int>(out_h) - mid_h) / 2;
	int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
	int left = (static_cast<int>(out_w) - mid_w) / 2;
	int right = (static_cast<int>(out_w) - mid_w + 1) / 2;
	if (dst.channels() == 1)
		cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
	cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	std::vector<float> pad_info{ static_cast<float>(left), static_cast<float>(top), scale };
	return pad_info;
}

// 不同导出器导出的 YOLO 输出维度顺序并不完全一致，这里归一化成 [N, C] 便于统一后处理
torch::Tensor DeepLearning_AI_pose::normalizeYoloOutput(torch::Tensor pred)
{
	if (pred.dim() == 3)
	{
		if (pred.size(0) == 1)
		{
			pred = pred.squeeze(0);
		}
		else if (pred.size(2) < pred.size(1))
		{
			pred = pred.transpose(1, 2).squeeze(0);
		}
	}

	if (pred.dim() == 2 && pred.size(0) < pred.size(1))
	{
		pred = pred.transpose(0, 1);
	}

	return pred;
}

// 将统一后的检测张量回填成旋转框、类别和关键点列表
void DeepLearning_AI_pose::appendPoseDetections(const torch::Tensor& det, const PoseLayout& layout, float pad_w, float pad_h, float scale,
	const cv::Rect& rectSrc, std::vector<cv::RotatedRect>& boxes, std::string& scores,
	std::string& classes, std::vector<cv::Point2f>& points)
{
	const int kpt_start = 6;
	for (int index = 0; index < det.size(0); ++index) {
		boxes.push_back(tensorBoxToRotatedRect(det[index], pad_w, pad_h, scale, rectSrc));
		scores.append(to_string(det[index][4].item().toFloat())).append(",");
		classes.append(to_string(det[index][5].item().toInt())).append(",");

		for (int kpt_index = 0; kpt_index < layout.num_kps; ++kpt_index) {
			const int base_index = kpt_start + kpt_index * layout.kpt_dim;
			if (base_index + 1 >= det.size(1)) {
				break;
			}

			float kpx = (det[index][base_index].item().toFloat() - pad_w) / scale + rectSrc.x;
			float kpy = (det[index][base_index + 1].item().toFloat() - pad_h) / scale + rectSrc.y;
			points.push_back(cv::Point2f(kpx, kpy));

			float kpt_score = 1.0f;
			if (layout.kpt_dim >= 3 && base_index + 2 < det.size(1)) {
				kpt_score = det[index][base_index + 2].item().toFloat();
			}
			m_pPoseKptScore.append(to_string(kpt_score)).append(",");
		}
	}
}

// 将单个检测框从 letterbox 坐标系还原到原图坐标系，并包装成 OpenCV 旋转框
cv::RotatedRect DeepLearning_AI_pose::tensorBoxToRotatedRect(const torch::Tensor& det_row, float pad_w, float pad_h, float scale, const cv::Rect& rectSrc)
{
	float x1 = (det_row[0].item().toFloat() - pad_w) / scale;
	float y1 = (det_row[1].item().toFloat() - pad_h) / scale;
	float x2 = (det_row[2].item().toFloat() - pad_w) / scale;
	float y2 = (det_row[3].item().toFloat() - pad_h) / scale;

	cv::RotatedRect rect;
	rect.center.x = (x1 + x2) / 2 + rectSrc.x;
	rect.center.y = (y1 + y2) / 2 + rectSrc.y;
	rect.size.width = x2 - x1;
	rect.size.height = y2 - y1;
	rect.angle = 0;
	return rect;
}


// CPU 版 NMS 核心实现
// 输入必须已经在 CPU 上，返回保留下来的行索引
at::Tensor DeepLearning_AI_pose::nms_kernel(
	const at::Tensor& dets,
	const at::Tensor& scores,
	float iou_threshold) {
	TORCH_CHECK(!dets.is_cuda(), "dets must be a CPU tensor");
	TORCH_CHECK(!scores.is_cuda(), "scores must be a CPU tensor");
	TORCH_CHECK(
		dets.scalar_type() == scores.scalar_type(),
		"dets should have the same type as scores");

	if (dets.numel() == 0)
		return at::empty({ 0 }, dets.options().dtype(at::kLong));

	auto x1_t = dets.select(1, 0).contiguous();
	auto y1_t = dets.select(1, 1).contiguous();
	auto x2_t = dets.select(1, 2).contiguous();
	auto y2_t = dets.select(1, 3).contiguous();

	at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

	auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

	auto ndets = dets.size(0);
	at::Tensor suppressed_t = at::zeros({ ndets }, dets.options().dtype(at::kByte));
	at::Tensor keep_t = at::zeros({ ndets }, dets.options().dtype(at::kLong));

	auto suppressed = suppressed_t.data_ptr<uint8_t>();
	auto keep = keep_t.data_ptr<int64_t>();
	auto order = order_t.data_ptr<int64_t>();
	auto x1 = x1_t.data_ptr<float>();
	auto y1 = y1_t.data_ptr<float>();
	auto x2 = x2_t.data_ptr<float>();
	auto y2 = y2_t.data_ptr<float>();
	auto areas = areas_t.data_ptr<float>();

	int64_t num_to_keep = 0;

	for (int64_t _i = 0; _i < ndets; _i++) {
		auto i = order[_i];
		if (suppressed[i] == 1)
			continue;
		keep[num_to_keep++] = i;
		auto ix1 = x1[i];
		auto iy1 = y1[i];
		auto ix2 = x2[i];
		auto iy2 = y2[i];
		auto iarea = areas[i];

		for (int64_t _j = _i + 1; _j < ndets; _j++) {
			auto j = order[_j];
			if (suppressed[j] == 1)
				continue;
			auto xx1 = std::max(ix1, x1[j]);
			auto yy1 = std::max(iy1, y1[j]);
			auto xx2 = std::min(ix2, x2[j]);
			auto yy2 = std::min(iy2, y2[j]);

			auto w = std::max(static_cast<float>(0), xx2 - xx1);
			auto h = std::max(static_cast<float>(0), yy2 - yy1);
			auto inter = w * h;
			auto ovr = inter / (iarea + areas[j] - inter);
			if (ovr > iou_threshold)
				suppressed[j] = 1;
		}
	}
	return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}


// 解析处理阶段 XML，将本次调用的输入图、ROI、阈值和调试开关同步到实例成员
int DeepLearning_AI_pose::parseProcessXml(void* inImg, char* xmlIn, string apiName)
{
	try
	{
		ResolveXml rXml;
		// 初始化图像数据
		m_pSrcImg = *((cv::Mat*)inImg);

		m_pVecRotaRect.clear();
		rXml.parseXmlRectROI(xmlIn, "Param", apiName, "rotRectROI", m_pVecRotaRect);
		//rXml.parseXmlRect(xmlIn, "Param", apiName, "searchRectROI", m_pRect);
		m_pDebugProcess = rXml.parseXmlInt(xmlIn, "Param", apiName, "debugProcess");
		m_pDebugResult = rXml.parseXmlInt(xmlIn, "Param", apiName, "debugResult");

		m_pThreshold = static_cast<float>(rXml.parseXmlDou(xmlIn, "Param", apiName, "confThreshold"));

	}
	catch (const std::exception& e)
	{
		CString str;
		str.Format("parseProcessXml std Exception %s", e.what());
		DeepLearning_AI_pose::addAiLog(LOG_ERROR, str.GetBuffer());
		return -1;
	}

	return 0;
}

// 统一前向推理入口：按当前后端执行 forward，并将输出整理成 tensor 列表
// 1. ONNX 路径返回 ORT 输出转换后的张量
// 2. TorchScript 路径兼容 Tensor / Tuple 两种输出形式
std::vector<torch::Tensor> DeepLearning_AI_pose::forwardWithTiming(const torch::Tensor& tensor_img)
{
	auto start = std::chrono::high_resolution_clock::now();

	if (m_modelRuntimeFormat == ModelRuntimeFormat::Onnx)
	{
		auto outputs = forwardOnnx(tensor_img);
		logStageCost("inference", start);
		return outputs;
	}

	torch::jit::IValue output = m_pModule.forward({ tensor_img });
	logStageCost("inference", start);

	std::vector<torch::Tensor> outputs;
	if (output.isTensor())
	{
		outputs.push_back(output.toTensor());
	}
	else if (output.isTuple())
	{
		for (const auto& element : output.toTuple()->elements())
		{
			outputs.push_back(element.toTensor());
		}
	}
	else
	{
		throw std::runtime_error("Unsupported TorchScript output type.");
	}
	return outputs;
}




