#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/torch.h>
#include <mutex>
#include <memory>
#include <chrono>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "../_common_files/renderParam.h"

using namespace std;

class DeepLearning_OCR
{

public:
	// OCR 预处理结果
	struct OcrInput
	{
		torch::Tensor tensor;			// 模型输入张量
		int resized_width;				// 等比例缩放后的图像宽度
		int input_width;				// 模型实际输入宽度
		int input_height;				// 模型实际输入高度
	};

public:
	// 模块运行参数
	bool							m_pDebugProcess = false;								// 是否保存调试过程
	bool							m_pDebugResult = false;									// 是否保存调试结果
	bool							m_pIsLoadedModule = false;								// 模型是否已加载
	bool							m_pModStatus = false;									// 模块状态
	int								m_ocrInputWidth = 320;									// OCR 模型输入宽度
	int								m_ocrInputHeight = 48;									// OCR 模型输入高度
	int8_t							gpuID = 0;												// GPU 设备编号
	// 图像及 ROI 数据
	cv::Mat							m_pSrcImg;												// 当前输入源图像
	cv::Mat							m_pDrcImg;												// OCR 识别区域图像
	cv::Rect						m_pRect;												// 当前矩形 ROI
	vector<cv::RotatedRect>			m_pVecRotaRect;											// 旋转矩形 ROI 集合
	// OCR 识别参数
	bool							m_ocrUseSpaceChar = true;								// 空格字符识别
	string							m_ocrDictPath;											// OCR 字符字典路径
	vector<string>					m_ocrCharacters;										// OCR 字符表
	float							m_pThreshold = 0.5;										// OCR 结果置信度阈值
	// 模型后端参数
	string							m_pModulePath;											// 模型文件路径
	int								m_pDeviceType = 0;										// 推理设备类型，0->CPU,1->GPU
	std::mutex						m_pMutex;												// 模型推理互斥锁
	// ONNX Runtime 参数
	unique_ptr<Ort::Env>			m_ortEnv;												// ONNX Runtime 环境
	unique_ptr<Ort::Session>		m_ortSession;											// ONNX Runtime 会话
	vector<string>					m_ortInputNames;										// ONNX 模型输入节点名称
	vector<string>					m_ortOutputNames;										// ONNX 模型输出节点名称
	bool							m_ortUsingCuda = false;									// ONNX Runtime 是否使用 CUDA
	int								m_ortCudaDeviceId = 0;									// ONNX Runtime CUDA 设备编号
	renderParam						m_renderparam;

public:
	static void configureOcrLoggingForRequest(const char* xmlIn, const std::string& apiName);
	static bool addOcrLog(long logTypeCode, const char* content);
	int loadModule(char* xmlIn, string apiName);
	void warmup(at::Tensor img);
	vector<torch::Tensor> forwardWithTiming(const torch::Tensor& tensor_img);
	int parseProcessXml(void* inImg, char* xmlIn, string apiName);
	int initOcrObj(char* xmlIn, char** xmlOut, string apiName);
	int parseInitOcrXml(char* xmlIn, string apiName);
	int processOneImgByOcr(void* inImg, void*& outImg, char* xmlIn, char** xmlOut, string apiName);
	int destoryOcrObject(char* xmlIn, char** xmlOut, string apiName);
	int outputOcrImg(cv::Mat inImg, void*& outImg);
	int runOcr(cv::Mat& m_pSrcImg, string& ocrText, float& ocrScore);
	int outputInitXml(char* xmlOut, string apiName);
	void outputDesXml(char* xmlOut, string apiName);
	void rotateRect2Rect();
	void judgeROI(const cv::Mat imgIn);

private:
	// 判断模型路径是否为 ONNX 文件。
	bool isOnnxModelPath(const string& module_path);
	// 重置模型加载状态、模块状态和 OCR 字典缓存，供重新加载模型时使用。
	void resetRuntimeState();
	// ONNX 模型加载与会话初始化。
	int loadOnnxModule();
	// ONNX 预热推理。
	void warmupOnnx(const torch::Tensor& img);
	// 将 Torch Tensor 封装为 ONNX Runtime 输入并执行推理，返回所有输出 Tensor。
	vector<torch::Tensor> forwardOnnx(const torch::Tensor& tensor_img);
	// 清理 ONNX 会话相关资源。
	void resetOnnxRuntimeState();
	// 若 ONNX 模型声明了静态输入尺寸，则用模型尺寸覆盖 XML 中的宽高配置。
	void updateOcrInputSizeFromOnnxSession(int& inputWidth, int& inputHeight) const;
	// 输出单个处理阶段耗时，便于定位预处理、推理或后处理瓶颈。
	void logStageCost(const char* stage_name, std::chrono::high_resolution_clock::time_point start);
	// 将 YOLO 输出框从中心点宽高格式转换为左上右下格式，保留兼容检测类后处理。
	void xywhToXyxyInPlace(torch::Tensor& pred);
	// 统一 YOLO 输出张量布局，保留兼容检测类后处理。
	torch::Tensor normalizeYoloOutput(torch::Tensor pred);
	// OCR 字典只在首次识别时加载一次
	void ensureOcrDictionaryLoaded();
	// 按 PaddleOCR 识别模型要求构造输入张量：等比例缩放、归一化、右侧补零、NHWC 转 NCHW。
	OcrInput prepareOcrInput(const cv::Mat& image);
	// greedy CTC 解码，折叠 blank 和连续重复字符，返回识别文本和平均分数。
	pair<string, float> decodeOcrPrediction(const torch::Tensor& output) const;
};
