#include "pch.h"
#include "DeepLearning_OCR.h"
#include "../_common_files/FileLogEx.h"
#include "../_common_files/ResolveXml.h"
#include "../_common_files/renderParam.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cctype>
#include <cmath>

// moduleID -> OCR对象，全局接口通过它复用已初始化模型。
std::unordered_map<int, DeepLearning_OCR*> g_mapDLE_ocr;

// OCR 日志与 AI_tool 保持一致，按线程内调试开关决定是否真正落盘。
CFileLogEx g_log;
std::once_flag g_ocrLoggerInitFlag;

namespace
{
	// 单次接口调用内生效，避免把一个请求的调试开关泄露到其他线程。
	thread_local bool ocrLoggingEnabled = false;

	void ensureOcrLoggerInitialized()
	{
		if (!ocrLoggingEnabled)
			return;

		std::call_once(g_ocrLoggerInitFlag, []()
			{
				char logPath[] = "d:\\debug\\log";
				g_log.initLog(logPath, "DeepLearning_ocr.log");
				g_log.Open(logPath, "DeepLearning_ocr.log");
			});
	}
}

void DeepLearning_OCR::configureOcrLoggingForRequest(const char* xmlIn, const std::string& apiName)
{
	ocrLoggingEnabled = false;
	if (xmlIn == nullptr)
		return;

	ResolveXml xml;
	const int debugProcess = xml.parseXmlInt(xmlIn, "Param", apiName, "debugProcess");
	const int debugResult = xml.parseXmlInt(xmlIn, "Param", apiName, "debugResult");
	ocrLoggingEnabled = debugProcess == 1 || debugResult == 1;
}

bool DeepLearning_OCR::addOcrLog(long logTypeCode, const char* content)
{
	if (!ocrLoggingEnabled || content == nullptr)
		return false;

	ensureOcrLoggerInitialized();
	return g_log.AddLog(logTypeCode, content);
}

namespace
{
	// 下面这些工具函数只服务 OCR 的 XML 输出、调试图保存和后端适配。
	renderParam makeRenderParam(const std::vector<cv::RotatedRect>& rects, const std::vector<cv::Point2f>& dots);
	void beginProcessXml(CMarkup& xml, DeepLearning_OCR* deepAI, const std::string& apiName, const char* moduleStatusCh);
	void finishProcessXml(CMarkup& xml, char** xmlOut);
	void addScoreCategoryXml(CMarkup& xml, const std::string& score, const std::string& category, bool trimTrailingComma);
	void addAxisXml(CMarkup& xml, const std::vector<cv::RotatedRect>& rects, bool useRectAngle, const char* shortAxisName, const char* longAxisName);
	void addRenderXml(CMarkup& xml, renderParam& rendPar, const char* rectCh, bool includeDots);
	std::string trimTrailingCommaOrDefault(const std::string& value, const char* defaultValue);
	std::string utf8ToLocalString(const std::string& value);
	std::string localToUtf8String(const std::string& value);

	std::string toLowerCopy(const std::string& value)
	{
		std::string lower = value;
		std::transform(lower.begin(), lower.end(), lower.begin(),
			[](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
		return lower;
	}

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

	DeepLearning_OCR* findDeepAIByModuleXml(char* xmlIn, const std::string& apiName)
	{
		ResolveXml rXml;
		int id = rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");
		auto mapIter = g_mapDLE_ocr.find(id);
		if (mapIter == g_mapDLE_ocr.end())
			return nullptr;

		return mapIter->second;
	}

	void prepareProcessXmlInput(DeepLearning_OCR* deepAI, void* inImg, char* xmlIn, const std::string& apiName)
	{
		int ret = deepAI->parseProcessXml(inImg, xmlIn, apiName);
		if (ret)
			throw "parseProcessXml Error";

		deepAI->rotateRect2Rect();
	}

	void saveDebugSourceImage(DeepLearning_OCR* deepAI)
	{
		if (!deepAI->m_pDebugProcess)
			return;

		cv::Mat buf = deepAI->m_pSrcImg.clone();
		cv::rectangle(buf, deepAI->m_pRect, cv::Scalar(255, 0, 0), 2, 8);
		cv::imwrite("d:\\debug\\spring_Src.jpg", buf);
	}

	void saveDebugResultImage(
		DeepLearning_OCR* deepAI,
		const std::vector<cv::RotatedRect>& rects)
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

		cv::imwrite("d:\\debug\\spring_Result.jpg", buf);
	}

	bool pathExists(const std::filesystem::path& path)
	{
		std::error_code ec;
		return !path.empty() && std::filesystem::exists(path, ec);
	}

	cv::Mat cropByPreparedRectRoi(DeepLearning_OCR* deepAI, cv::Rect& rectSrc)
	{
		// prepareProcessXmlInput 已经通过 rotateRect2Rect() 把首个旋转 ROI 折算到 m_pRect。
		rectSrc = deepAI->m_pRect;
		return deepAI->m_pSrcImg(rectSrc);
	}

	cv::RotatedRect rectToRotatedRect(const cv::Rect& rect)
	{
		return cv::RotatedRect(
			cv::Point2f(
				static_cast<float>(rect.x) + static_cast<float>(rect.width) * 0.5f,
				static_cast<float>(rect.y) + static_cast<float>(rect.height) * 0.5f),
			cv::Size2f(static_cast<float>(rect.width), static_cast<float>(rect.height)),
			0.0f);
	}

	void writeOcrResultXml(
		char** xmlOut,
		DeepLearning_OCR* deepAI,
		const std::string& apiName,
		int recStatus,
		const std::vector<cv::RotatedRect>& rects,
		const std::string& text,
		float score)
	{
		CMarkup xml;
		renderParam rendPar = makeRenderParam(rects, {});
		CString scoreStr;
		scoreStr.Format("%.6f", score);

		beginProcessXml(xml, deepAI, apiName, "模块状态");
		xml.AddElem("recStatus", recStatus);
		xml.SetAttrib("CH", "结果：识别到");
		xml.AddElem("renderRectBox", rendPar.conRenderRect().c_str());
		xml.SetAttrib("CH", "识别区域");
		const std::string xmlText = utf8ToLocalString(text);
		xml.AddElem("recText", xmlText.c_str());
		xml.SetAttrib("CH", "识别内容");
		xml.AddElem("score", scoreStr.GetBuffer());
		xml.SetAttrib("CH", "置信度分数");
		finishProcessXml(xml, xmlOut);
	}

	void writeOcrNormalResultXml(
		char** xmlOut,
		DeepLearning_OCR* deepAI,
		const std::string& apiName,
		int recStatus,
		const std::vector<cv::RotatedRect>& rects)
	{
		CMarkup xml;
		renderParam rendPar = makeRenderParam(rects, {});

		beginProcessXml(xml, deepAI, apiName, "模块状态");
		xml.AddElem("recStatus", recStatus);
		xml.SetAttrib("CH", "结果：未识别到");
		xml.AddElem("renderRectBox", rendPar.conRenderRect().c_str());
		xml.SetAttrib("CH", "识别区域");
		xml.AddElem("recText", "");
		xml.SetAttrib("CH", "识别内容");
		xml.AddElem("score", 0);
		xml.SetAttrib("CH", "置信度分数");
		finishProcessXml(xml, xmlOut);
	}

	void logWholeProcessCost(DWORD startTick)
	{
		DWORD endTick = GetTickCount64();
		string str = "The whole process Running Time is :";
		str.append(to_string(endTick - startTick));
		DeepLearning_OCR::addOcrLog(LOG_MESSAGE, str.c_str());
	}

	renderParam makeRenderParam(const std::vector<cv::RotatedRect>& rects, const std::vector<cv::Point2f>& dots)
	{
		renderParam rendPar;
		rendPar.m_renderRectBox = rects;
		rendPar.m_renderDot = dots;
		return rendPar;
	}

	void beginProcessXml(CMarkup& xml, DeepLearning_OCR* deepAI, const std::string& apiName, const char* moduleStatusCh)
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

	void finishProcessXml(CMarkup& xml, char** xmlOut)
	{
		xml.OutOfElem();
		CString Out = xml.GetDoc();
		const std::string utf8Xml = localToUtf8String(std::string(Out.GetBuffer()));
		strcpy(*xmlOut, utf8Xml.c_str());
	}

	void addScoreCategoryXml(CMarkup& xml, const std::string& score, const std::string& category, bool trimTrailingComma)
	{
		const std::string scoreText = trimTrailingComma ? trimTrailingCommaOrDefault(score, "0") : score;
		const std::string categoryText = trimTrailingComma ? trimTrailingCommaOrDefault(category, "-100") : category;

		xml.AddElem("score", scoreText.c_str());
		xml.SetAttrib("CH", "置信度置信度分数");
		xml.AddElem("category", categoryText.c_str());
		xml.SetAttrib("CH", "类别");
	}

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
		xml.AddElem("Angle", angleStr.c_str());
		xml.SetAttrib("CH", "角度");
	}

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

	std::string trimTrailingCommaOrDefault(const std::string& value, const char* defaultValue)
	{
		if (value.empty())
			return defaultValue;
		return value.substr(0, value.length() - 1);
	}

	std::string bytesToHex(const std::string& value)
	{
		static const char* hexDigits = "0123456789ABCDEF";
		std::string hex;
		hex.reserve(value.size() * 2);
		for (const unsigned char ch : value)
		{
			hex.push_back(hexDigits[(ch >> 4) & 0x0F]);
			hex.push_back(hexDigits[ch & 0x0F]);
		}
		return hex;
	}

	std::wstring utf8ToWide(const std::string& value)
	{
		if (value.empty())
		{
			return L"";
		}

		const int wideLen = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, nullptr, 0);
		if (wideLen <= 0)
		{
			return L"";
		}

		std::vector<wchar_t> wideBuf(static_cast<size_t>(wideLen));
		if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, wideBuf.data(), wideLen) <= 0)
		{
			return L"";
		}

		return std::wstring(wideBuf.data());
	}

	std::string wideToCodePage(const std::wstring& value, UINT codePage)
	{
		if (value.empty())
		{
			return "";
		}

		const int byteLen = WideCharToMultiByte(codePage, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
		if (byteLen <= 0)
		{
			return "";
		}

		std::vector<char> byteBuf(static_cast<size_t>(byteLen));
		if (WideCharToMultiByte(codePage, 0, value.c_str(), -1, byteBuf.data(), byteLen, nullptr, nullptr) <= 0)
		{
			return "";
		}

		return std::string(byteBuf.data());
	}

	std::string utf8ToLocalString(const std::string& value)
	{
		const std::wstring wide = utf8ToWide(value);
		if (wide.empty())
		{
			return value;
		}
		const std::string local = wideToCodePage(wide, CP_ACP);
		return local.empty() ? value : local;
	}

	std::string localToUtf8String(const std::string& value)
	{
		if (value.empty())
		{
			return "";
		}

		const int wideLen = MultiByteToWideChar(CP_ACP, 0, value.c_str(), -1, nullptr, 0);
		if (wideLen <= 0)
		{
			return value;
		}

		std::vector<wchar_t> wideBuf(static_cast<size_t>(wideLen));
		if (MultiByteToWideChar(CP_ACP, 0, value.c_str(), -1, wideBuf.data(), wideLen) <= 0)
		{
			return value;
		}

		const std::string utf8 = wideToCodePage(std::wstring(wideBuf.data()), CP_UTF8);
		return utf8.empty() ? value : utf8;
	}
}

bool DeepLearning_OCR::isOnnxModelPath(const std::string& module_path)
{
	const std::string extension = toLowerCopy(std::filesystem::path(module_path).extension().string());
	return extension == ".onnx";
}

void DeepLearning_OCR::resetRuntimeState()
{
	m_pIsLoadedModule = false;
	m_pModStatus = false;
	m_ocrCharacters.clear();
	resetOnnxRuntimeState();
}

int DeepLearning_OCR::loadOnnxModule()
{
	m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DeepLearning_OCR");
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

	return 0;
}

void DeepLearning_OCR::resetOnnxRuntimeState()
{
	m_ortSession.reset();
	m_ortEnv.reset();
	m_ortInputNames.clear();
	m_ortOutputNames.clear();
	m_ortUsingCuda = false;
	m_ortCudaDeviceId = 0;
}

int DeepLearning_OCR::initOcrObj(char* xmlIn, char** xmlOut, string apiName)
{
	int ret = 0;
	int id = 0;
	DeepLearning_OCR* deepAI = nullptr;

	id = parseInitOcrXml(xmlIn, apiName);

	auto mapIter = g_mapDLE_ocr.find(id);
	if (mapIter != g_mapDLE_ocr.end())
		deepAI = mapIter->second;
	else
	{
		// 首次出现的 moduleID 在这里创建实例，后续重复初始化走复用。
		deepAI = new DeepLearning_OCR;
		g_mapDLE_ocr.insert(std::pair<int, DeepLearning_OCR*>(id, deepAI));
	}

	ret = deepAI->loadModule(xmlIn, apiName);
	deepAI->outputInitXml(*xmlOut, apiName);

	return ret;
}

int DeepLearning_OCR::parseInitOcrXml(char* xmlIn, std::string apiName)
{
	ResolveXml rXml;
	return rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");
}

int DeepLearning_OCR::outputInitXml(char* xmlOut, string apiName)
{
	CMarkup xml;
	xml.SetDoc("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n");
	xml.AddElem("Param");
	xml.IntoElem();

	xml.AddElem(apiName.c_str());
	xml.IntoElem();
	CString initNumStr;

	int modStatus = m_pModStatus ? 1 : 0;
	initNumStr.Format("%d", modStatus);
	xml.AddElem("moduleStatus", initNumStr.GetBuffer());
	xml.SetAttrib("CH", "模块状态");

	xml.OutOfElem();
	CString Out = xml.GetDoc();
	strcpy(xmlOut, Out.GetBuffer());
	return 0;
}

int DeepLearning_OCR::destoryOcrObject(char* xmlIn, char** xmlOut, std::string apiName)
{
	DeepLearning_OCR* deepAI = nullptr;
	ResolveXml rXml;
	int id = rXml.parseXmlInt(xmlIn, "Param", apiName, "moduleID");

	auto mapIter = g_mapDLE_ocr.find(id);
	if (mapIter == g_mapDLE_ocr.end())
		return -4;

	deepAI = mapIter->second;
	deepAI->m_pModStatus = false;
	deepAI->outputDesXml(*xmlOut, apiName);
	delete deepAI;
	g_mapDLE_ocr.erase(mapIter);
	return 0;

}

void DeepLearning_OCR::outputDesXml(char* xmlOut, std::string apiName)
{
	CMarkup xml;
	xml.SetDoc("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n");
	xml.AddElem("Param");
	xml.IntoElem();

	xml.AddElem(apiName.c_str());
	xml.IntoElem();
	CString initNumStr;

	int desStatu = m_pModStatus ? 1 : 0;
	initNumStr.Format("%d", desStatu);
	xml.AddElem("moduleStatus", initNumStr.GetBuffer());
	xml.SetAttrib("CH", "模块状态");

	xml.OutOfElem();
	CString Out = xml.GetDoc();
	strcpy(xmlOut, Out.GetBuffer());
}

//输出OCR识别图片
int DeepLearning_OCR::outputOcrImg(cv::Mat inImg, void*& outImg)
{
	int size = inImg.total() * inImg.elemSize();
	unsigned char* dstMat = new unsigned char[size];
	std::memcpy(dstMat, inImg.data, size * sizeof(unsigned char));
	outImg = new cv::Mat(inImg.rows, inImg.cols, inImg.type(), dstMat);

	return 0;
}

int DeepLearning_OCR::loadModule(char* xmlIn, string apiName)
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

		m_ocrInputWidth = rXml.parseXmlInt(xmlIn, "Param", apiName, "inputWidth");
		const int parsedOcrInputHeight = rXml.parseXmlInt(xmlIn, "Param", apiName, "inputHeight");
		const int parsedUseSpaceChar = rXml.parseXmlInt(xmlIn, "Param", apiName, "useSpaceChar");
		std::string ocrDictPath = rXml.parseXmlStr(xmlIn, "Param", apiName, "characterDictPath");
		if (ocrDictPath.empty())
		{
			ocrDictPath = rXml.parseXmlStr(xmlIn, "Param", apiName, "recCharDictPath");
		}
		if (!ocrDictPath.empty())
		{
			m_ocrDictPath = ocrDictPath;
			m_ocrCharacters.clear();
		}
		if (parsedOcrInputHeight > 0)
		{
			m_ocrInputHeight = parsedOcrInputHeight;
		}
		if (parsedUseSpaceChar == 0 || parsedUseSpaceChar == 1)
		{
			m_ocrUseSpaceChar = parsedUseSpaceChar == 0;
		}
		if (!strcmp(modulePath.c_str(), m_pModulePath.c_str()))
		{
			// 模型路径没变且当前实例可用时，直接复用已加载模型。
			if (m_pIsLoadedModule && m_pModStatus)
				return 0;
		}
		resetRuntimeState();
		m_pModulePath = modulePath;

		torch::set_num_threads(2);

		if (m_ocrInputWidth == 0)
		{
			m_ocrInputWidth = 320;
		}

		if (!isOnnxModelPath(m_pModulePath))
		{
			throw std::runtime_error(
				"Unsupported model file extension. DeepLearning_OCR only accepts .onnx models.");
		}

		loadOnnxModule();
		
		int warmupHeight = m_ocrInputHeight > 0 ? m_ocrInputHeight : 48;
		int warmupWidth = m_ocrInputWidth > 0 ? m_ocrInputWidth : 320;

		// ONNX 若声明了静态输入尺寸，预热阶段也要和 session 保持一致。
		//updateOcrInputSizeFromOnnxSession(warmupWidth, warmupHeight);
		
		// OCR 常用“高固定、宽自适应”的输入约束，因此 warmup 形状与检测类模型不同。
		std::vector<int64_t> warmupShape = { 1, 3, static_cast<int64_t>(warmupHeight), static_cast<int64_t>(warmupWidth) };
		
		auto img = torch::rand(warmupShape);
		std::cout << "img size: " << img.sizes() << std::endl;
		DeepLearning_OCR::warmup(img);

		std::string backendMessage;
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
		std::cout << backendMessage << std::endl;

	}
	catch (const std::exception& e)
	{
		std::cerr << "loadModule std Exception: " << e.what() << std::endl;
		CString str;
		str.Format("loadModule std Exception %s", e.what());
		DeepLearning_OCR::addOcrLog(LOG_ERROR, str.GetBuffer());
		m_pIsLoadedModule = false;
		m_pModStatus = false;
		return -1;
	}

	m_pIsLoadedModule = true;
	m_pModStatus = true;
	return 0;
}

void DeepLearning_OCR::warmup(at::Tensor img)
{
	auto start = std::chrono::high_resolution_clock::now();
	warmupOnnx(img);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "warmup cost : " << duration.count() << " ms" << std::endl;

}

void DeepLearning_OCR::warmupOnnx(const torch::Tensor& img)
{
	(void)forwardOnnx(img);
	(void)forwardOnnx(img);
}

std::vector<torch::Tensor> DeepLearning_OCR::forwardOnnx(const torch::Tensor& tensor_img)
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

int DeepLearning_OCR::processOneImgByOcr(void* inImg, void*& outImg, char* xmlIn, char** xmlOut, std::string apiName)
{
	DeepLearning_OCR* deepAI = findDeepAIByModuleXml(xmlIn, apiName);
	if (deepAI == nullptr)
		return -4;

	try
	{
		DWORD t1 = GetTickCount64();

		// 解析输入图、ROI 和调试参数，并把旋转 ROI 折成普通矩形使用。
		prepareProcessXmlInput(deepAI, inImg, xmlIn, apiName);
		saveDebugSourceImage(deepAI);

		cv::Rect rectSrc;
		cv::Mat cjImg = cropByPreparedRectRoi(deepAI, rectSrc);

		std::string ocrText;
		float ocrScore = 0.0f;
		int result = 0;
		
		result = deepAI->runOcr(cjImg, ocrText, ocrScore);
		
		std::vector<cv::RotatedRect> renderRects = deepAI->m_pVecRotaRect;
		if (renderRects.empty())
		{
			renderRects.push_back(rectToRotatedRect(cv::Rect(0, 0, deepAI->m_pSrcImg.cols, deepAI->m_pSrcImg.rows)));
		}

		//deepAI->m_pDrcImg = deepAI->m_pSrcImg.clone();
		//for (const auto& rect : renderRects)
		//{
		//	cv::Point2f pts[4];
		//	rect.points(pts);
		//	std::vector<cv::Point> contour;
		//	contour.reserve(4);
		//	for (const auto& pt : pts)
		//	{
		//		contour.emplace_back(cvRound(pt.x), cvRound(pt.y));
		//	}
		//	cv::polylines(deepAI->m_pDrcImg, contour, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
		//	if (result > 0 && !contour.empty())
		//	{
		//		// 当前 OCR 仅输出一段文本，直接贴在结果框首点附近。
		//		cv::putText(
		//			deepAI->m_pDrcImg,
		//			ocrText.empty() ? "" : ocrText,
		//			contour.front(),
		//			cv::FONT_HERSHEY_SIMPLEX,
		//			0.7,
		//			cv::Scalar(0, 255, 0),
		//			2,
		//			cv::LINE_AA);
		//	}
		//}
		//// 输出图像构造
		//m_renderparam.conOutImg(deepAI->m_pDrcImg, outImg);

		int DC_run = 2;
		if (result > 0)
		{
			DC_run = 1;
			saveDebugResultImage(deepAI, renderRects);
			writeOcrResultXml(xmlOut, deepAI, apiName, DC_run, renderRects, ocrText, ocrScore);
		}
		else
		{
			writeOcrNormalResultXml(xmlOut, deepAI, apiName, DC_run, renderRects);
		}

		logWholeProcessCost(t1);
	}
	catch (const std::exception& e)
	{
		CString str;
		str.Format("processOneImg OCR std Exception %s", e.what());
		DeepLearning_OCR::addOcrLog(LOG_ERROR, str.GetBuffer());
		return -2;
	}
	catch (cv::Exception& e)
	{
		CString str;
		str.Format("processOneImg OCR cv Exception %s", e.what());
		DeepLearning_OCR::addOcrLog(LOG_ERROR, str.GetBuffer());
		return -3;
	}

	return 0;
}

int DeepLearning_OCR::runOcr(cv::Mat& m_pSrcImg, std::string& ocrText, float& ocrScore)
{
	torch::NoGradGuard no_grad;
	ensureOcrDictionaryLoaded();

	auto preStart = std::chrono::high_resolution_clock::now();
	OcrInput input = prepareOcrInput(m_pSrcImg);
	logStageCost("pre-process ", preStart);

	auto outputs = forwardWithTiming(input.tensor);
	if (outputs.empty())
	{
		return 0;
	}

	auto postStart = std::chrono::high_resolution_clock::now();
	const auto decodeResult = decodeOcrPrediction(outputs[0]);
	ocrText = decodeResult.first;
	ocrScore = decodeResult.second;

	if (m_pThreshold > 0.0f && ocrScore < m_pThreshold)
	{
		ocrText.clear();
		ocrScore = 0.0f;
	}
	logStageCost("post-process", postStart);

	return ocrText.empty() ? 0 : 1;
}
void DeepLearning_OCR::rotateRect2Rect()
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

void DeepLearning_OCR::logStageCost(const char* stage_name, std::chrono::high_resolution_clock::time_point start)
{
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << stage_name << " takes : " << duration.count() * 1e-3 << " ms" << std::endl;
}

int DeepLearning_OCR::parseProcessXml(void* inImg, char* xmlIn, string apiName)
{
	try
	{
		ResolveXml rXml;
		m_pSrcImg = *((cv::Mat*)inImg);

		m_pVecRotaRect.clear();
		rXml.parseXmlRectROI(xmlIn, "Param", apiName, "rotRectROI", m_pVecRotaRect);
		m_pDebugProcess = rXml.parseXmlInt(xmlIn, "Param", apiName, "debugProcess");
		m_pDebugResult = rXml.parseXmlInt(xmlIn, "Param", apiName, "debugResult");
		m_pThreshold = static_cast<float>(rXml.parseXmlDou(xmlIn, "Param", apiName, "ocrConfThreshold"));

	}
	catch (const std::exception& e)
	{
		CString str;
		str.Format("parseProcessXml std Exception %s", e.what());
		DeepLearning_OCR::addOcrLog(LOG_ERROR, str.GetBuffer());
		return -1;
	}

	return 0;
}

std::vector<torch::Tensor> DeepLearning_OCR::forwardWithTiming(const torch::Tensor& tensor_img)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto outputs = forwardOnnx(tensor_img);
	logStageCost("inference", start);
	return outputs;
}

void DeepLearning_OCR::ensureOcrDictionaryLoaded()
{
	if (!m_ocrCharacters.empty())
	{
		return;
	}

	const std::string dictPath = m_ocrDictPath;
	if (dictPath.empty())
	{
		throw std::runtime_error("OCR character dictionary file was not found.");
	}

	std::ifstream dictFile(std::filesystem::path(dictPath), std::ios::binary);
	if (!dictFile.is_open())
	{
		throw std::runtime_error("Failed to open OCR character dictionary file.");
	}

	m_ocrCharacters.clear();
	std::string line;
	//仅删除第一行真实 UTF-8 BOM，避免字符表索引错位。
	bool firstLine = true;
	while (std::getline(dictFile, line))
	{
		if (firstLine && line.size() >= 3 &&
			static_cast<unsigned char>(line[0]) == 0xEF &&
			static_cast<unsigned char>(line[1]) == 0xBB &&
			static_cast<unsigned char>(line[2]) == 0xBF)
		{
			line.erase(0, 3);
		}
		firstLine = false;
		while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
		{
			line.pop_back();
		}
		if (!line.empty())
		{
			m_ocrCharacters.push_back(line);
		}
	}

	if (m_ocrUseSpaceChar)
	{
		m_ocrCharacters.push_back(" ");
	}

	if (m_ocrCharacters.empty())
	{
		throw std::runtime_error("OCR character dictionary is empty.");
	}

	m_ocrDictPath = dictPath;
}

DeepLearning_OCR::OcrInput DeepLearning_OCR::prepareOcrInput(const cv::Mat& image)
{
	cv::Mat bgr = image.clone();
	if (bgr.empty())
	{
		throw std::runtime_error("OCR input image is empty.");
	}
	if (bgr.channels() == 1)
	{
		cv::cvtColor(bgr, bgr, cv::COLOR_GRAY2BGR);
	}
	else if (bgr.channels() == 4)
	{
		cv::cvtColor(bgr, bgr, cv::COLOR_BGRA2BGR);
	}

	int inputHeight = m_ocrInputHeight > 0 ? m_ocrInputHeight : 48;
	int inputWidth = m_ocrInputWidth > 0 ? m_ocrInputWidth : 320;
	updateOcrInputSizeFromOnnxSession(inputWidth, inputHeight);

	// OCR 宽度按比例自适应，再限制到模型最大输入宽。
	const float ratio = static_cast<float>(bgr.cols) / static_cast<float>(std::max(1, bgr.rows));
	int resizedWidth = static_cast<int>(std::ceil(static_cast<float>(inputHeight) * ratio));
	resizedWidth = std::max(1, std::min(inputWidth, resizedWidth));

	cv::Mat resized;
	cv::resize(bgr, resized, cv::Size(resizedWidth, inputHeight), 0, 0, cv::INTER_LINEAR);
	resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
	resized = (resized - cv::Scalar(0.5, 0.5, 0.5)) / 0.5;

	// Pad after normalization so the right-side blank area stays at 0, matching PaddleOCR.
	cv::Mat chwReady(inputHeight, inputWidth, CV_32FC3, cv::Scalar(0, 0, 0));
	resized.copyTo(chwReady(cv::Rect(0, 0, resizedWidth, inputHeight)));

	// NHWC -> NCHW.
	torch::Tensor tensor = torch::from_blob(
		chwReady.data,
		{ 1, inputHeight, inputWidth, 3 },
		torch::kFloat32).clone();
	tensor = tensor.permute({ 0, 3, 1, 2 }).contiguous();

	return { tensor, resizedWidth, inputWidth, inputHeight };
}

// 从 ONNX Runtime Session 读取模型输入的 NCHW shape，并用模型声明的静态尺寸
// 更新 inputWidth / inputHeight。
// - shape[2]（高度）> 0 时，覆盖 inputHeight；否则保留调用方传入值。
// - shape[3]（宽度）> 0 时，覆盖 inputWidth；否则保留调用方传入值。
// - shape[3]（宽度）动态维度通常为 -1，不会覆盖 XML 配置的尺寸。
void DeepLearning_OCR::updateOcrInputSizeFromOnnxSession(int& inputWidth, int& inputHeight) const
{
	if (!m_ortSession)
	{
		return;
	}

	auto typeInfo = m_ortSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
	const std::vector<int64_t> shape = typeInfo.GetShape();
	if (shape.size() < 4)
	{
		return;
	}

	if (shape[2] > 0)
	{
		inputHeight = static_cast<int>(shape[2]);
	}
	if (shape[3] > 0)   //shape[3]为模型自适应width -> 动态维度（-1），不覆盖
	{
		inputWidth = static_cast<int>(shape[3]);
	}
}

std::pair<std::string, float> DeepLearning_OCR::decodeOcrPrediction(const torch::Tensor& output) const
{
	torch::Tensor pred = output.detach().to(torch::kCPU).to(torch::kFloat32);
	if (pred.dim() == 2)
	{
		pred = pred.unsqueeze(0);
	}
	if (pred.dim() != 3)
	{
		throw std::runtime_error("Unsupported OCR output shape.");
	}

	const int64_t dictSizeWithBlank = static_cast<int64_t>(m_ocrCharacters.size()) + 1;
	if (pred.size(1) == dictSizeWithBlank && pred.size(2) != dictSizeWithBlank)
	{
		// 兼容 [N, C, T] 与 [N, T, C] 两种 OCR 输出布局。
		pred = pred.transpose(1, 2);
	}

	torch::Tensor probTensor = pred;
	auto maxResult = torch::max(probTensor, 2);
	torch::Tensor indices = std::get<1>(maxResult).to(torch::kInt64);
	torch::Tensor probs = std::get<0>(maxResult).to(torch::kFloat32);

	std::string text;
	float probSum = 0.0f;
	int validCount = 0;
	int64_t lastIndex = -1;
	const int64_t timeSteps = indices.size(1);

	for (int64_t step = 0; step < timeSteps; ++step)
	{
		const int64_t index = indices[0][step].item<int64_t>();
		if (index == 0)
		{
			lastIndex = 0;
			continue;
		}
		if (index == lastIndex)
		{
			continue;
		}

		// blank 跳过、连续重复折叠，保持 CTC greedy decode 语义。
		const size_t charIndex = static_cast<size_t>(index - 1);
		if (charIndex < m_ocrCharacters.size())
		{
			text.append(m_ocrCharacters[charIndex]);
			probSum += probs[0][step].item<float>();
			++validCount;
		}
		lastIndex = index;
	}

	const float meanScore = validCount > 0 ? probSum / static_cast<float>(validCount) : 0.0f;
	return { text, meanScore };
}
