#include "pch.h"
#include "DeepLearning_AI_segment.h"
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

// 全局模型对象映射表：moduleID → DeepLearning_AI_segment 实例指针
// DLL 内所有推理流程通过此表查找已注册的模型对象
std::unordered_map<int, DeepLearning_AI_segment*> g_mapDLE_AI_segment;

// 全局日志对象，用于记录所有操作日志
extern CFileLogEx g_log;
extern std::once_flag g_aiLoggerInitFlag;

namespace
{
	thread_local bool segmentAiLoggingEnabled = false;

	void ensureSegmentAiLoggerInitialized()
	{
		if (!segmentAiLoggingEnabled)
			return;

		std::call_once(g_aiLoggerInitFlag, []()
			{
				char logPath[] = "d:\\debug\\log";
				g_log.initLog(logPath, "DeepLearning_AI_segment.log");
				g_log.Open(logPath, "DeepLearning_AI_segment.log");
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
		hint.append(", but this TorchScript segment model appears to expect inputSize=");
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

void DeepLearning_AI_segment::configureAiLoggingForRequest(const char* xmlIn, const std::string& apiName)
{
	segmentAiLoggingEnabled = false;
	if (xmlIn == nullptr)
		return;

	ResolveXml xml;
	const int debugProcess = xml.parseXmlInt(xmlIn, "Param", apiName, "debugProcess");
	const int debugResult = xml.parseXmlInt(xmlIn, "Param", apiName, "debugResult");
	segmentAiLoggingEnabled = debugProcess == 1 || debugResult == 1;
}

bool DeepLearning_AI_segment::addAiLog(long logTypeCode, const char* content)
{
	if (!segmentAiLoggingEnabled || content == nullptr)
		return false;

	ensureSegmentAiLoggerInitialized();
	return g_log.AddLog(logTypeCode, content);
}

namespace
{
	renderParam makeRenderParam(
		const std::vector<cv::RotatedRect>& rects,
		const std::vector<std::vector<cv::Point2f>>& polygons,
		const std::vector<cv::Point2f>& dots);
	void beginProcessXml(CMarkup& xml, DeepLearning_AI_segment* deepAI, const std::string& apiName, const char* moduleStatusCh);
	void finishProcessXml(CMarkup& xml, char** xmlOut);
	void addScoreCategoryXml(CMarkup& xml, const std::string& score, const std::string& category, bool trimTrailingComma);
	void addAxisXml(CMarkup& xml, const std::vector<cv::RotatedRect>& rects, bool useRectAngle, const char* shortAxisName, const char* longAxisName);
	void addRenderXml(CMarkup& xml, renderParam& rendPar, const char* rectCh, bool includeDots);
	std::string trimTrailingCommaOrDefault(const std::string& value, const char* defaultValue);

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
	DeepLearning_AI_segment* findDeepAIByModuleXml(char* xmlIn, const std::string& apiName)
	{
		ResolveXml rXml;
		int id = rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");
		auto mapIter = g_mapDLE_AI_segment.find(id);
		if (mapIter == g_mapDLE_AI_segment.end())
			return nullptr;

		return mapIter->second;
	}

	// 解析处理阶段 XML，并把 rotRectROI 统一折算成普通矩形 ROI，供后续裁剪与回填坐标使用
	void prepareProcessXmlInput(DeepLearning_AI_segment* deepAI, void* inImg, char* xmlIn, const std::string& apiName)
	{
		int ret = deepAI->parseProcessXml(inImg, xmlIn, apiName);
		if (ret)
			throw "parseProcessXml Error";

		deepAI->rotateRect2Rect();
	}

	// 调试模式下保存源图与 ROI，方便定位 XML 传入范围是否正确
	void saveDebugSourceImage(DeepLearning_AI_segment* deepAI)
	{
		if (!deepAI->m_pDebugProcess)
			return;

		cv::Mat buf = deepAI->m_pSrcImg.clone();
		cv::rectangle(buf, deepAI->m_pRect, cv::Scalar(255, 0, 0), 2, 8);
		cv::imwrite("d:\\debug\\spring_Src.jpg", buf);
	}

	// 调试模式下保存分割 mask 图，方便直接查看 runSegment 的输出
	void saveDebugResultImage(
		DeepLearning_AI_segment* deepAI,
		const cv::Mat& maskImg)
	{
		if (!deepAI->m_pDebugResult)
			return;
		if (maskImg.empty())
			return;

		cv::imwrite("d:\\debug\\spring_Result.jpg", maskImg);
	}

	// 若调用方传入 rotRectROI，则截取首个 ROI 对应子图；否则直接使用整图
	cv::Mat cropByPreparedRectRoi(DeepLearning_AI_segment* deepAI, cv::Rect& rectSrc)
	{
		// prepareProcessXmlInput 已经通过 rotateRect2Rect() 把首个旋转 ROI 折算到 m_pRect
		rectSrc = deepAI->m_pRect;
		return deepAI->m_pSrcImg(rectSrc);
	}


	// 统一生成“检测到目标/缺陷”的结果 XML
	void writeObjectResultXml(
		char** xmlOut,
		DeepLearning_AI_segment* deepAI,
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
		renderParam rendPar = makeRenderParam(rects, {}, dots);

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
		DeepLearning_AI_segment* deepAI,
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
			xml.SetAttrib("CH", "渲染轮廓点集");
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

	// 分割路径沿用历史逻辑：先转三通道，再根据矩形 ROI 裁剪
	cv::Mat cropSegmentInput(DeepLearning_AI_segment* deepAI)
	{
		cv::Mat matIn = deepAI->m_pSrcImg.clone();
		if (matIn.channels() != 3)
		{
			cv::cvtColor(matIn, matIn, cv::COLOR_GRAY2RGB);
		}
		deepAI->judgeROI(matIn);
		return matIn(deepAI->m_pRect);
	}

	// 输出统一的整条调用耗时，便于从 DLL 日志观察端到端性能
	void logWholeProcessCost(DWORD startTick)
	{
		DWORD endTick = GetTickCount64();
		string str = "The whole process Running Time is :";
		str.append(to_string(endTick - startTick));
		DeepLearning_AI_segment::addAiLog(LOG_MESSAGE, str.c_str());
	}

	// 将矩形框与点集打包成 renderParam，统一供 XML 输出函数复用
	renderParam makeRenderParam(
		const std::vector<cv::RotatedRect>& rects,
		const std::vector<std::vector<cv::Point2f>>& polygons,
		const std::vector<cv::Point2f>& dots)
	{
		renderParam rendPar;
		rendPar.m_renderRectBox = rects;
		rendPar.m_renderPolygon = polygons;
		rendPar.m_renderDot = dots;
		return rendPar;
	}

	// 创建 XML 根结构并写入公共的模块状态字段
	void beginProcessXml(CMarkup& xml, DeepLearning_AI_segment* deepAI, const std::string& apiName, const char* moduleStatusCh)
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

	cv::RotatedRect normalizeLongEdgeRect(const cv::RotatedRect& srcRect)
	{
		cv::RotatedRect rect = srcRect;
		if (rect.size.width < rect.size.height)
		{
			std::swap(rect.size.width, rect.size.height);
			rect.angle += 90.0f;
		}
		while (rect.angle < 0.0f)
			rect.angle += 180.0f;
		while (rect.angle >= 180.0f)
			rect.angle -= 180.0f;
		return rect;
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
		//将矩形框序列化成字符串
		xml.AddElem("renderRectBox", rendPar.conRenderRect().c_str());
		xml.SetAttrib("CH", rectCh);

		if (includeDots)
		{
			//将点集序列化成字符串
			xml.AddElem("renderDot", rendPar.conRenderDot().c_str());
			xml.SetAttrib("CH", "渲染轮廓点集");
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
bool DeepLearning_AI_segment::isOnnxModelPath(const std::string& module_path)
{
	const std::string extension = toLowerCopy(std::filesystem::path(module_path).extension().string());
	return extension == ".onnx";
}

// 通过扩展名判断是否为 TorchScript 模型文件
bool DeepLearning_AI_segment::isTorchScriptModelPath(const std::string& module_path)
{
	const std::string extension = toLowerCopy(std::filesystem::path(module_path).extension().string());
	return extension == ".torchscript";
}

// 根据模型文件扩展名选择具体推理后端
bool DeepLearning_AI_segment::inferModelRuntimeFormat(const std::string& module_path, ModelRuntimeFormat& runtime_format)
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
void DeepLearning_AI_segment::resetRuntimeState()
{
	m_pIsLoadedModule = false;
	m_pModStatus = false;
	m_pModule = torch::jit::Module();
	m_modelRuntimeFormat = ModelRuntimeFormat::TorchScript;
	resetOnnxRuntimeState();
}

// 初始化 ONNXRuntime 会话，并缓存输入/输出名字，后续推理时不再重复查询元数据
int DeepLearning_AI_segment::loadOnnxModule()
{
	m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DeepLearning_AI_segment");
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
void DeepLearning_AI_segment::resetOnnxRuntimeState()
{
	m_ortSession.reset();
	m_ortEnv.reset();
	m_ortInputNames.clear();
	m_ortOutputNames.clear();
	m_ortUsingCuda = false;
	m_ortCudaDeviceId = 0;
}

// ============================================================================
// 分割模型初始化入口
// 流程：
//   1. 从 XML 解析 moduleID
//   2. 若全局 map 中不存在该 ID，则创建新的 CDeepAI 实例并插入 map
//   3. 调用 loadModule 加载模型文件
//   4. 将初始化状态输出为 XML
// 返回值：0=成功，非 0=失败
// ============================================================================
int DeepLearning_AI_segment::initSegmentAiObj(char* xmlIn, char** xmlOut, string apiName)
{
	int ret = 0;
	int id = 0;
	std::string modulePath;


	DeepLearning_AI_segment* deepAI = nullptr;

	id = parseInitSegmentXml(xmlIn, apiName);

	// 如果全局映射表中不存在该 ID，则创建新对象并插入映射表
	auto mapIter = g_mapDLE_AI_segment.find(id);
	if (mapIter != g_mapDLE_AI_segment.end())
		deepAI = mapIter->second;
	else
	{
		deepAI = new DeepLearning_AI_segment;
		g_mapDLE_AI_segment.insert(std::pair<int, DeepLearning_AI_segment*>(id, deepAI));
	}

	//deepAI->m_pMutex.lock();
	ret = deepAI->loadModule(xmlIn, apiName);

	deepAI->outputInitXml(*xmlOut, apiName);
	//deepAI->m_pMutex.unlock();

	return ret;
}

// parseInitXml — 从初始化 XML 中解析出 moduleID
int DeepLearning_AI_segment::parseInitSegmentXml(char* xmlIn, std::string apiName)
{
	int ret = 0;
	ResolveXml rXml;

	return rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");
}

// 生成模型初始化结果 XML，只包含模块状态字段
int DeepLearning_AI_segment::outputInitXml(char* xmlOut, string apiName)
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
//   2. 在全局 map 中查找并删除 DeepLearning_AI_segment 实例
//   3. 调用 outputDesXml 输出销毁状态 XML
// 注意：
// 返回值：0=成功，-4=模型未找到
// ============================================================================
int DeepLearning_AI_segment::destorySegmentObject(char* xmlIn, char** xmlOut, std::string apiName)
{
	DeepLearning_AI_segment* deepAI = nullptr;
	ResolveXml rXml;
	int id = rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");

	// 在全局 map 中查找目标模型实例
	auto mapIter = g_mapDLE_AI_segment.find(id);
	if (mapIter == g_mapDLE_AI_segment.end())
		return -4;

	deepAI = mapIter->second;
	deepAI->m_pModStatus = false;
	// 释放模型对象内存
	// 注意：此处未从 g_mapDLE_AI_segment 中 erase 该条目，后续访问会产生悬空指针；
	// 实际使用时需确保 destory 后不再引用该 moduleID
	deepAI->outputDesXml(*xmlOut, apiName);
	delete deepAI;
	g_mapDLE_AI_segment.erase(mapIter);
	return 0;

}

// outputDesXml — 输出销毁操作的 XML 状态报告
// 参数 xmlOut：输出 XML 缓冲区（由调用方预分配）
// 参数 apiName：XML 标签名
// 参数 desStatus：销毁是否成功（true→1, false→0）
void DeepLearning_AI_segment::outputDesXml(char* xmlOut, std::string apiName)
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

// 将 cv::Mat 深拷贝到 DLL 可对外传递的 void* 指针
// 策略：分配新内存并 memcpy 图像数据，使得调用方可以独立释放
int DeepLearning_AI_segment::outputSegmentImg(cv::Mat inImg, void*& outImg)
{
	// 计算图像数据总字节数并分配独立内存块
	int size = inImg.total() * inImg.elemSize();
	unsigned char* dstMat = new unsigned char[size];
	std::memcpy(dstMat, inImg.data, size * sizeof(unsigned char));
	// 用独立内存块构造新的 cv::Mat，确保生命周期独立于源图
	outImg = new cv::Mat(inImg.rows, inImg.cols, inImg.type(), dstMat);

	return 0;
}

// ============================================================================
// 解析初始化配置、加载模型后端，并完成预热
// 流程：
//   1. 解析 XML，读取 deviceType、modulePath、gpuID、YOLO 版本和输入尺寸
//   2. 若模型路径未变化且当前实例已处于可用状态，则直接复用已加载模型
//   3. 重置上一次加载留下的运行时状态，并记录新的模型路径
//   4. 若 inputSize 未显式指定(==0)，则按任务类型补默认输入尺寸
//   5. 根据模型文件扩展名选择 ONNXRuntime 或 TorchScript 后端，并执行实际加载
//   6. 构造分割模型 warmup 输入张量
//   7. 执行 warmup，并输出当前推理后端信息
// ============================================================================
int DeepLearning_AI_segment::loadModule(char* xmlIn, string apiName)
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

		// 没有显式传 inputSize 时，补分割模型的默认输入尺寸
		if (m_inputSize == 0) {

			m_inputSize = 640; // 分割
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

		// 分割路径使用固定方形 warmup 输入
		auto img = torch::rand({ 1, 3, m_inputSize, m_inputSize }).to(m_pDevice);
		std::cout << "img size: " << img.sizes() << std::endl;
		// 模型预热推理
		DeepLearning_AI_segment::warmup(img);

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

int DeepLearning_AI_segment::inferSegNcFromPreds(const torch::Tensor& preds)
{
	torch::Tensor pred = preds.squeeze();

	if (pred.dim() == 2 && pred.size(0) < pred.size(1))
	{
		pred = pred.transpose(0, 1);
	}

	const int nm = 32;
	const int nc = static_cast<int>(pred.size(-1)) - 4 - nm;

	return nc > 0 ? nc : 0;
}

int DeepLearning_AI_segment::inferSegNcFromTorchscriptOutput(const torch::jit::IValue& output)
{
	std::vector<torch::Tensor> outputs;
	// TorchScript 原始输出：torch::jit::IValue，需要拆 Tensor / Tuple
	if (output.isTensor())
	{
		outputs.push_back(output.toTensor());
	}
	else if (output.isTuple())
	{
		for (const auto& element : output.toTuple()->elements())
		{
			if (element.isTensor())
			{
				outputs.push_back(element.toTensor());
			}
		}
	}
	else
	{
		return 0;
	}

	if (outputs.empty())
		return 0;

	return inferSegNcFromPreds(outputs[0]);
}

// warmup — GPU/CUDA 预热推理
// 执行两次空 forward 以触发 CUDA kernel 编译和显存分配，
// 避免正式推理时因延迟初始化导致耗时波动
// 参数 img：与模型输入尺寸匹配的随机张量
void DeepLearning_AI_segment::warmup(at::Tensor img)
{
	auto start = std::chrono::high_resolution_clock::now();

	if (m_modelRuntimeFormat == ModelRuntimeFormat::Onnx)
	{
		warmupOnnx(img);
	}
	else
	{
		//第一次 forward：触发模型初始化、算子加载、CUDA kernel 编译、显存分配等。
		torch::jit::IValue output = m_pModule.forward({ img });
		//第二次 forward：让后续正式推理更接近稳定耗时，避免第一次推理特别慢。
		torch::jit::IValue output1 = m_pModule.forward({ img });

		const int nc = inferSegNcFromTorchscriptOutput(output1);
		if (nc > 0)
		{
			buildSegClassGrayValues(nc);
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "warmup cost : " << duration.count() << " ms" << std::endl;

}

// ONNX 预热与 TorchScript 保持一致：执行两次空推理，尽量把初始化成本前置
void DeepLearning_AI_segment::warmupOnnx(const torch::Tensor& img)
{
	(void)forwardOnnx(img);
	//(void)forwardOnnx(img);
	std::vector<torch::Tensor> outputs = forwardOnnx(img);
	if (!outputs.empty())
	{
		const int nc = inferSegNcFromPreds(outputs[0]);
		if (nc > 0)
		{
			buildSegClassGrayValues(nc);
		}
	}
}

// 统一把输入拉回 CPU 连续内存再喂给 ONNXRuntime，避免不同设备/步长下的数据兼容问题
std::vector<torch::Tensor> DeepLearning_AI_segment::forwardOnnx(const torch::Tensor& tensor_img)
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

	//ONNX 原始输出：Ort::Value 列表
	std::vector<torch::Tensor> outputs;
	outputs.reserve(ort_outputs.size());
	for (auto& output : ort_outputs)
	{
		outputs.push_back(ortValueToTensor(output));
	}
	return outputs;
}

// ============================================================================
//分割任务处理主流程
// 流程：
//   1. 解析 XML 获取 moduleID 并查找对应 CDeepAI 实例
//   2. parseProcessXml：解析输入图像与 ROI 参数
//   3. rotateRect2Rect：旋转矩形→正矩形转换
//   4. judgeROI：ROI 边界保护
//   5. runSegment：LibTorch 分割推理 + 后处理
//   6. outputSegmentImg：将分割结果图像深拷贝为 DLL 输出格式
//   7. outputXml：将结果序列化为 XML
// 参数 inImg：输入图像（cv::Mat* 转为 void*）
// 参数 outImg：输出分割 掩膜图像（cv::Mat* 转为 void*&）
// 参数 xmlIn：推理配置 XML
// 参数 xmlOut：输出结果 XML（char**，DLL 内部已分配缓冲区）
// 参数 apiName：XML 标签名
// 返回值：0=成功，-2=标准异常，-3=OpenCV 异常，-4=模型未找到
// ============================================================================
int DeepLearning_AI_segment::processOneImgBySegment(void* inImg, void*& outImg, char* xmlIn, char** xmlOut, std::string apiName)
{
	DeepLearning_AI_segment* deepAI = findDeepAIByModuleXml(xmlIn, apiName);
	if (deepAI == nullptr)
		return -4;

	//deepAI->m_pMutex.lock();
	try
	{
		DWORD t1 = GetTickCount64();
		prepareProcessXmlInput(deepAI, inImg, xmlIn, apiName);
		saveDebugSourceImage(deepAI);
		cv::Mat curImg = cropSegmentInput(deepAI);

		cv::Mat m_pDstImg(curImg.rows, curImg.cols, CV_8U, cv::Scalar(0));
		std::string DC_score, DC_class;

		int result = 0;

		result = deepAI->runSegment(curImg, m_pDstImg, DC_score, DC_class);// 结果包含检测框、置信度分数和类别索引

		renderParam rendPar;
		rendPar.m_renderRectBox.clear();
		rendPar.m_renderPolygon.clear();
		rendPar.m_renderDot.clear();

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(m_pDstImg.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (const auto& contour : contours)
		{
			if (contour.empty())
				continue;

			double area = std::fabs(cv::contourArea(contour));
			if (area <= 0.0)
				continue;
			std::vector<cv::Point2f> polygon;
			polygon.reserve(contour.size());
			for (const auto& pt : contour)
			{
				const cv::Point2f renderPt(
					static_cast<float>(pt.x + deepAI->m_pRect.x),
					static_cast<float>(pt.y + deepAI->m_pRect.y));
				polygon.push_back(renderPt);
				rendPar.m_renderDot.push_back(renderPt);
			}
			// 轮廓点先平移回原图坐标，再计算最小外接矩形，并统一成“长边方向角”输出
			const cv::RotatedRect normalizedRect = normalizeLongEdgeRect(cv::minAreaRect(polygon));
			rendPar.m_renderRectBox.push_back(normalizedRect);
			rendPar.m_renderPolygon.push_back(std::move(polygon));
		}

		saveDebugResultImage(deepAI, m_pDstImg);

		int DC_run = 1;
		if (result > 0 && !rendPar.m_renderRectBox.empty())
		{
			DC_run = 2;
			writeObjectResultXml(
				xmlOut,
				deepAI,
				apiName,
				DC_run,
				rendPar.m_renderRectBox,
				rendPar.m_renderDot,
				DC_score,
				DC_class,
				true,
				true,
				"shortAxis",
				"longAxis",
				"模块状态",
				"渲染矩形");
		}
		else
		{
			writeNormalResultXml(xmlOut, deepAI, apiName, DC_run, true, true, "shortAxis", "longAxis");
		}

		//outputSegmentImg(m_pDstImg, outImg);

		// 输出图像构造
		m_renderparam.conOutImg(m_pDstImg, outImg);

		logWholeProcessCost(t1);

	}
	catch (const std::exception& e)
	{
		std::cerr << "processOneImg std Exception: " << e.what() << std::endl;
		CString str;
		str.Format("processOneImg std Exception %s", e.what());
		DeepLearning_AI_segment::addAiLog(LOG_ERROR, str.GetBuffer());

		//deepAI->m_pMutex.unlock();
		return -2;
	}
	catch (cv::Exception& e)
	{
		CString str;
		str.Format("processOneImg cv Exception %s", e.what());
		DeepLearning_AI_segment::addAiLog(LOG_ERROR, str.GetBuffer());

		//deepAI->m_pMutex.unlock();
		return -3;
	}
	//deepAI->m_pMutex.unlock();
	return 0;
}

// ============================================================================
// runSegment — 分割模型推理与后处理
// 流程：
//   1. 预处理：Letterbox 缩放填充 → OpenCV Mat → LibTorch BCHW 张量
//   2. 推理：m_pModule.forward() → 获取 preds 和 proto 输出
//   3. 后处理：PostProcessing_seg 生成分割 mask 到 m_pDstImg
// 返回值：检测到的目标数量（0 表示无结果）
// ============================================================================
int DeepLearning_AI_segment::runSegment(cv::Mat& m_pSrcImg, cv::Mat& m_pDstImg, std::string& DC_score, std::string& DC_class) {

	float conf_threshold = m_pThreshold; float iou_threshold = 0.45;
	int image_size = m_inputSize;
	torch::NoGradGuard no_grad;
	// TODO：检查图像尺寸

	/*** 预处理 ***/
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<float> pad_info = LetterboxImage(m_pSrcImg, m_pSrcImg, cv::Size(image_size, image_size)); // 缩放并填充输入图像

	//std::vector<float> CDeepAI::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size)
	const float pad_w = pad_info[0]; // 获取横向填充宽度
	const float pad_h = pad_info[1];  // 获取纵向填充高度
	const float scale = pad_info[2];   // 获取缩放比例


	//将 OpenCV 图像数据转换为 PyTorch 张量格式
	auto tensor_img = torch::from_blob(m_pSrcImg.data, { 1, m_pSrcImg.rows, m_pSrcImg.cols, m_pSrcImg.channels() }, torch::kUInt8).to(m_pDevice);//.to(torch::Device(torch::DeviceType::CUDA, gpuID));// 许双龙
	// 将张量图像数据转换为 PyTorch BCHW 格式，并设置为连续内存
	tensor_img = tensor_img.flip(3).permute({ 0, 3, 1, 2 }).div(255).contiguous();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	// 首次执行通常耗时更长
	std::cout << "pre-process takes : " << duration.count() * 1e-3 << " ms" << std::endl;

	// 推理
	start = std::chrono::high_resolution_clock::now();
	auto outputs = forwardWithTiming(tensor_img);
	if (outputs.size() < 2)
	{
		throw std::runtime_error("Segmentation model must return two tensors.");
	}
	at::Tensor preds = outputs[0];
	at::Tensor proto = outputs[1].cpu()[0];

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	// 首次执行通常耗时更长
	std::cout << "inference takes : " << duration.count() * 1e-3 << " ms " << std::endl;

	int num_det = PostProcessing_seg(preds, proto, m_pDstImg, pad_w, pad_h, scale, m_pSrcImg.size(), conf_threshold, iou_threshold, DC_score, DC_class);
	auto end2 = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end2 - end);
	std::cout << "post takes : " << duration.count() * 1e-3 << " ms " << std::endl;

	return num_det;
}

// 将首个 rotRectROI 转成普通矩形
// 若未配置 ROI，则默认整图参与推理
void DeepLearning_AI_segment::rotateRect2Rect()
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

// 输出单阶段耗时，阶段名由调用方传入
void DeepLearning_AI_segment::logStageCost(const char* stage_name, std::chrono::high_resolution_clock::time_point start)
{
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << stage_name << " takes : " << duration.count() * 1e-3 << " ms" << std::endl;
}

// 通用 letterbox 预处理：（保持原图比例不变的缩放 + 补边预处理方式）
// 保持长宽比缩放后居中补边，同时返回 pad_w/pad_h/scale 供坐标反算使用
// ============================================================================
// 参数 src：输入图像（只读）
// 参数 dst：输出图像（会被原地修改）
// 参数 out_size：目标尺寸（通常为正方形，如 640×640）
// 返回值：{pad_left, pad_top, scale} 三个浮点数
// ============================================================================
std::vector<float> DeepLearning_AI_segment::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
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

// CPU 版 NMS 核心实现
// 输入必须已经在 CPU 上，返回保留下来的行索引
at::Tensor DeepLearning_AI_segment::nms_kernel(
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
int DeepLearning_AI_segment::parseProcessXml(void* inImg, char* xmlIn, string apiName)
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
		DeepLearning_AI_segment::addAiLog(LOG_ERROR, str.GetBuffer());
		return -1;
	}

	return 0;
}

// 统一前向推理入口：按当前后端执行 forward，并将输出整理成 tensor 列表
// 1. ONNX 路径返回 ORT 输出转换后的张量
// 2. TorchScript 路径兼容 Tensor / Tuple 两种输出形式
std::vector<torch::Tensor> DeepLearning_AI_segment::forwardWithTiming(const torch::Tensor& tensor_img)
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

// ============================================================================
// 分割模型推理与后处理
// 流程：
//   1. 预处理：Letterbox 缩放填充 → OpenCV Mat → LibTorch BCHW 张量
//   2. 推理：m_pModule.forward() → 获取 preds 和 proto 输出
//   3. 后处理：PostProcessing_seg 生成分割 mask 到 m_pDstImg
// 参数 m_pSrcImg：输入图像（会被 Letterbox 原地修改）
// 参数 m_pDstImg：输出分割 掩膜图像
// 返回值：检测到的目标数量（0 表示无结果）
// ============================================================================
void DeepLearning_AI_segment::judgeROI(const cv::Mat imgIn)
{
	if (m_pRect.x < 0)
	{
		m_pRect.width += m_pRect.x;
		m_pRect.x = 0;
	}

	if (m_pRect.x + m_pRect.width > imgIn.cols)
	{
		m_pRect.width = imgIn.cols - m_pRect.x;
	}

	if (m_pRect.y < 0)
	{
		m_pRect.height += m_pRect.y;
		m_pRect.y = 0;
	}

	if (m_pRect.y + m_pRect.height > imgIn.rows)
	{
		m_pRect.height = imgIn.rows - m_pRect.y;
	}
}

// 将二维张量浅包装为 Mat，不做拷贝，调用方需保证底层张量生命周期覆盖使用期
cv::Mat DeepLearning_AI_segment::tensor2Mat(const torch::Tensor& input_tensor) {
	int height = input_tensor.size(0), width = input_tensor.size(1);
	cv::Mat o_Mat(cv::Size(width, height), CV_32F, input_tensor.data_ptr());
	return o_Mat;
}

// 分割后处理：将原型掩膜与检测框组合，恢复到图像空间并绘制到输出掩膜mask图中
// 流程：
//   1. 置信度过滤 → (cx,cy,w,h) → (x1,y1,x2,y2)
//   2. 得分×类别概率 → NMS 去重
//   3. einsum 生成每个检测的 mask 原型 → sigmoid 激活
//   4. 将 mask 裁剪、缩放后叠加到 m_pDstImg（不同类别用不同灰度值区分）
// 参数 preds：模型输出的预测张量
// 参数 proto：模型输出的 mask 原型张量
// 参数 m_pDstImg：输出分割 掩膜mask图像
// 返回值：检测到的目标数量（0 表示无结果）
int DeepLearning_AI_segment::PostProcessing_seg(const torch::Tensor& preds, const torch::Tensor& proto, cv::Mat& m_pDstImg,
	float pad_w, float pad_h, float scale, const cv::Size& m_pSrcImg_shape,
	float conf_thres, float iou_thres, std::string& DC_score, std::string& DC_class)
{
	// 导出格式兼容：
	// 某些模型导出后是 [C, N]，某些是 [N, C]，这里统一整理成 [N, C](N 候选框数量， C 候选框携带的特征数量(bbox 4项 + 类别置信度分数 + mask系数))
	torch::Tensor pred = preds.squeeze();
	if (pred.dim() == 2 && pred.size(0) < pred.size(1)) {
		pred = pred.transpose(0, 1);
	}

	// 分割头输出结构约定：[x, y, w, h] + [nc 个类别置信度分数] + [nm 个 mask 系数]
	const int nm = 32;   //nm 表示每个目标用于生成分割 mask 的系数数量（YOLOv8 segmentation 导出模型里, mask prototype 的通道数就是 32）
	int nc = pred.size(-1) - 4 - nm;
	if (nc <= 0) return 0;

	// 已在warmup阶段解出模型的类别数nc，故这里就不需要再构建灰度值映射表
	//buildSegClassGrayValues(nc);

	// 第 1 段：先做一次粗过滤
	// 按“最大类别置信度分数”筛掉明显无效的候选框，减少后续 NMS 和 mask 计算开销
	torch::Tensor cls_scores = pred.slice(1, 4, 4 + nc);
	torch::Tensor scores = std::get<0>(cls_scores.max(1));
	pred = pred.index({ scores > conf_thres });
	if (pred.size(0) == 0) return 0;

	// 第 2 段：为保留下来的候选框确定最终类别与最终得分
	// max_tuple:
	//   第一个张量是每个候选框的最大类别得分
	//   第二个张量是对应的类别索引
	cls_scores = pred.slice(1, 4, 4 + nc);
	std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(cls_scores, 1);

	// 将框格式从中心点表示 [cx, cy, w, h] 转成角点表示 [x1, y1, x2, y2]，
	// 后面的 NMS 与 ROI 裁剪都更适合使用角点坐标
	pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2).div(2);
	pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3).div(2);
	pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
	pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

	// 把 score / class / mask 系数拼到统一检测张量里：
	// [x1, y1, x2, y2, score, class, mask_coeffs...]
	torch::Tensor score_col = std::get<0>(max_tuple).unsqueeze(1);
	torch::Tensor class_col = std::get<1>(max_tuple).to(torch::kFloat32).unsqueeze(1);
	torch::Tensor coeff_cols = pred.slice(1, 4 + nc, 4 + nc + nm);

	torch::Tensor dets = torch::cat({ pred.slice(1, 0, 4), score_col, class_col, coeff_cols }, 1).cpu();

	// 第 3 段：NMS 去重，只保留高质量且互不重叠的目标框
	auto keep = nms_kernel(dets.slice(1, 0, 4), dets.select(1, 4), iou_thres);
	auto det = torch::index_select(dets, 0, keep);
	if (det.size(0) == 0) return 0;

	// 第 4 段：生成每个检测目标对应的 mask
	// det.slice(1, 6, 6 + nm) 是每个目标的 32 个 mask 系数，
	// proto 是全图共享的 32 张 mask 原型
	// einsum("ij,jkm->ikm") 的含义是：
	//   第 i 个目标的 mask = 该目标的系数 * 所有原型 mask 的线性组合
	// sigmoid 之后把 logits 压到 [0,1]，表示像素属于该目标的概率
	torch::Tensor masks = torch::einsum("ij,jkm->ikm", { det.slice(1, 6, 6 + nm), proto }).sigmoid();

	// 第 5 段：把原型 mask 的坐标系映射回 letterbox 后图像的有效区域
	// proto 的空间大小通常比输入图像更小，因此需要先把 padding 对应区域裁掉，
	// 再 resize 回当前输出图像尺寸
	const int proto_h = proto.size(1);
	const int proto_w = proto.size(2);
	const int image_w = std::max(1, m_pSrcImg_shape.width);
	const int image_h = std::max(1, m_pSrcImg_shape.height);
	const int crop_x = std::clamp(static_cast<int>(std::round(pad_w * proto_w / static_cast<float>(image_w))), 0, proto_w - 1);
	const int crop_y = std::clamp(static_cast<int>(std::round(pad_h * proto_h / static_cast<float>(image_h))), 0, proto_h - 1);
	const int resized_w = std::max(1, std::min(proto_w - crop_x, static_cast<int>(std::round(m_pDstImg.cols * scale * proto_w / static_cast<float>(image_w)))));
	const int resized_h = std::max(1, std::min(proto_h - crop_y, static_cast<int>(std::round(m_pDstImg.rows * scale * proto_h / static_cast<float>(image_h)))));

	for (int index = 0; index < det.size(0); ++index) {
		// 保存当前目标的置信度分数和类别，后续用于 XML 输出
		float score = det[index][4].item().toFloat();
		int class_idx = det[index][5].item().toInt();
		DC_score.append(to_string(score)).append(",");
		DC_class.append(to_string(class_idx)).append(",");

		// 把检测框从 letterbox 坐标系还原到当前输出图像坐标系
		int x1 = (det[index][0].item().toFloat() - pad_w) / scale;  // 横向填充
		int y1 = (det[index][1].item().toFloat() - pad_h) / scale;  // 纵向填充
		int x2 = (det[index][2].item().toFloat() - pad_w) / scale;  // 横向填充
		int y2 = (det[index][3].item().toFloat() - pad_h) / scale;  // 纵向填充

		// 边界保护，防止 ROI 越界导致 OpenCV 子图操作崩溃
		x1 = std::min(std::max(x1, 0), m_pDstImg.cols);
		y1 = std::min(std::max(y1, 0), m_pDstImg.rows);
		x2 = std::min(std::max(x2, 0), m_pDstImg.cols);
		y2 = std::min(std::max(y2, 0), m_pDstImg.rows);

		// 取出当前目标的 mask，裁掉 letterbox padding，再 resize 回输出图大小
		cv::Mat mat = tensor2Mat(masks[index]);
		cv::Mat Temp = mat(cv::Rect(crop_x, crop_y, resized_w, resized_h)).clone();
		cv::resize(Temp, Temp, cv::Size(m_pDstImg.cols, m_pDstImg.rows));
		cv::Rect  roi = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));

		cv::Mat uchar_mat;

		// 将概率 mask 二值化后叠加到输出图
		// 这里按类别给不同灰度值，便于一张 mask 图里区分不同类别 
		const int val =
			(class_idx >= 0 && class_idx < static_cast<int>(m_segClassGrayValues.size()))
			? m_segClassGrayValues[class_idx]
			: 255;

		cv::threshold(Temp(roi), uchar_mat, 0.5, val, cv::THRESH_BINARY);
		m_pDstImg(roi) += uchar_mat;
	}
	return det.size(0);
}

// 根据类别ID构建灰度值映射表
// 灰度映射示例：  
//   nc == 1 -> 255  
//   nc == 2 -> 255, 128  
//   nc == 3 -> 255, 170, 85 
void DeepLearning_AI_segment::buildSegClassGrayValues(int nc)
{
	nc = std::max(nc, 1);

	if (m_cachedSegNc == nc && static_cast<int>(m_segClassGrayValues.size()) == nc)
		return;

	m_segClassGrayValues.clear();
	m_segClassGrayValues.reserve(nc);

	for (int class_idx = 0; class_idx < nc; ++class_idx)
	{
		//类别 ID 越大，灰度越大
		//const int gray_level = static_cast<int>(std::round(255.0 * (class_idx + 1) / nc));
		//类别 ID 越大，灰度越小
		//根据类别ID 定义灰度赋值规则(小数部分 < 0.5：向下取整；小数部分 >= 0.5：向上取整)
		const int gray_level = static_cast<int>(std::round(255.0 * (nc - class_idx) / nc));
		const int val = std::clamp(gray_level, 1, 255);
		m_segClassGrayValues.push_back(val);
	}

	m_cachedSegNc = nc;
}






