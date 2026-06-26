#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <mutex>
#include <memory>
#include <chrono>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "../_common_files/renderParam.h"

using namespace std;

class DeepLearning_AI_detect
{

public:
	// 当前类同时支持 TorchScript 与 ONNXRuntime 两种推理后端
	enum class ModelRuntimeFormat
	{
		TorchScript,
		Onnx
	};

public:
	// 模块运行参数
	bool							m_pDebugProcess = false;								// 调试开关：控制过程图的输出
	bool							m_pDebugResult = false;									// 调试开关：控制结果图的输出
	bool							m_pIsLoadedModule = false;								// 模型状态：是否已加载、状态是否正常、是否请求 GPU
	bool							m_pModStatus = false;									// 模块状态
	int								m_inputSize = 640;										// 模型输入尺寸（默认检测用 640）
	int8_t							gpuID = 0;												// GPU 编号，TorchScript CUDA 和 ONNXRuntime CUDA 路径共用
	// 图像及 ROI 数据
	cv::Mat							m_pSrcImg;												// 当前输入源图像
	cv::Mat							m_pDrcImg;												// 当前输出结果图像
	cv::Rect						m_pRect;												// 当前生效的轴对齐矩形 ROI
	vector<cv::RotatedRect>			m_pVecRotaRect;											// 旋转矩形 ROI 相关结果集合
	// 模型后端参数
	torch::jit::Module				m_pModule;												// TorchScript 模型对象与执行设备
	c10::Device						m_pDevice = c10::Device(c10::kCPU);						// LibTorch 设备类型，默认 CPU
	ModelRuntimeFormat				m_modelRuntimeFormat = ModelRuntimeFormat::TorchScript; // 当前模型实际采用的推理后端
	string							m_pModulePath;											// 模型路径与公共推理参数
	int								m_pDeviceType = 0;										// 是否使用 GPU 推理
	float							m_pThreshold = 0.25;									// 检测时默认置信度阈值
	std::mutex						m_pMutex;												// 保护模型加载与推理过程的互斥锁
	// ONNX Runtime 参数
	unique_ptr<Ort::Env>			m_ortEnv;												// ONNX Runtime 环境
	unique_ptr<Ort::Session>		m_ortSession;											// ONNX Runtime 会话
	vector<string>					m_ortInputNames;										// ONNX 模型输入节点名称
	vector<string>					m_ortOutputNames;										// ONNX 模型输出节点名称
	bool							m_ortUsingCuda = false;									// ONNX Runtime 是否使用 CUDA
	int								m_ortCudaDeviceId = 0;									// ONNX Runtime CUDA 设备编号
	renderParam						m_renderparam;

public:
	static void configureAiLoggingForRequest(const char* xmlIn, const std::string& apiName);
	static bool addAiLog(long logTypeCode, const char* content);
	// 读取初始化 XML，加载模型，并完成一次 warmup
	int loadModule(char* xmlIn, std::string apiName);
	// 预热推理，用于提前触发模型初始化、算子加载和显存分配
	void warmup(at::Tensor img);
	// 统一前向推理入口：内部根据当前后端返回一个或多个张量输出
	std::vector<torch::Tensor> forwardWithTiming(const torch::Tensor& tensor_img);
	// 解析处理阶段 XML，将输入图像、ROI、阈值等信息写入成员变量
	int parseProcessXml(void* inImg, char* xmlIn, std::string apiName);
	// 从初始化 XML 中解析 moduleID
	int parseInitDetectXml(char* xmlIn, std::string apiName);
	// 执行单张图像的目标检测任务
	int processOneImgByDetection(void* inImg, void*& outImg, char* xmlIn, char** xmlOut, std::string apiName);
	// 初始化或复用一个 DeepLearning_AI_detect 实例，并输出初始化结果 XML
	int initDetectAiObj(char* xmlIn, char** xmlOut, std::string apiName);
	// 目标检测推理，输出轴对齐框及分数/类别
	int runDetection(cv::Mat& m_pSrcImg, std::vector<cv::RotatedRect>& DC_VecRotaRect, std::string& DC_score, std::string& DC_class, std::vector<cv::Point2f>& DC_Pts, cv::Rect& rectSrc);
	// 输出初始化结果 XML
	int outputInitXml(char* xmlOut, std::string apiName);
	// 按 moduleID 销毁实例并输出销毁结果 XML
	int destoryDetectObject(char* xmlIn, char** xmlOut, std::string apiName);
	// 生成销毁接口的返回 XML
	void outputDesXml(char* xmlOut, std::string apiName);
	// 输出检测图像结果
	int outputDetectImg(cv::Mat inImg, void*& outImg);
	// 将旋转框 ROI 转成普通矩形 ROI
	void rotateRect2Rect();

private:
	// 统一封装 YOLO 系列模型预处理结果：
	// 包括张量本体、letterbox 的 padding 和缩放系数
	struct InferenceInput
	{
		torch::Tensor tensor;
		float pad_w;
		float pad_h;
		float scale;
	};

	// 判断模型路径是否为 ONNX 文件
	bool isOnnxModelPath(const std::string& module_path);
	// 判断模型路径是否为 TorchScript 文件
	bool isTorchScriptModelPath(const std::string& module_path);
	// 根据模型路径扩展名推断实际运行后端
	bool inferModelRuntimeFormat(const std::string& module_path, ModelRuntimeFormat& runtime_format);
	// 清空运行时状态，避免重复加载不同模型时残留旧状态
	void resetRuntimeState();
	// ONNX 模型加载与会话初始化
	int loadOnnxModule();
	// ONNX 预热推理
	void warmupOnnx(const torch::Tensor& img);
	// ONNX 前向推理，并统一转成 torch::Tensor 结果
	std::vector<torch::Tensor> forwardOnnx(const torch::Tensor& tensor_img);
	// 清理 ONNX 会话相关资源
	void resetOnnxRuntimeState();
	// YOLO 路径统一预处理
	InferenceInput prepareYoloInput(cv::Mat& image, int image_size);
	// 记录单阶段耗时日志
	void logStageCost(const char* stage_name, std::chrono::high_resolution_clock::time_point start);
	// 原地将 [cx, cy, w, h] 转成 [x1, y1, x2, y2]
	void xywhToXyxyInPlace(torch::Tensor& pred);
	// 对不同 YOLO 导出格式做归一化，统一成后处理可接受的二维张量
	torch::Tensor normalizeYoloOutput(torch::Tensor pred);
	// 常规检测后处理：阈值过滤、类别选择与 NMS
	torch::Tensor buildDetectionDetections(const torch::Tensor& output, float conf_threshold, float iou_threshold);
	// 将单个检测张量转换为原图坐标系下的旋转框
	cv::RotatedRect tensorBoxToRotatedRect(const torch::Tensor& det_row, float pad_w, float pad_h, float scale, const cv::Rect& rectSrc);
	// 将轴对齐检测结果批量写入输出容器
	void appendAxisAlignedDetections(const torch::Tensor& det, float pad_w, float pad_h, float scale, const cv::Rect& rectSrc,
		std::vector<cv::RotatedRect>& boxes, std::string& scores, std::string& classes);
	// 执行 letterbox 缩放与补边，返回 padding 与缩放系数
	std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(640, 640));
	// NMS 核心实现，返回保留下来的索引集合
	at::Tensor nms_kernel(const at::Tensor& dets, const at::Tensor& scores, float iou_threshold);
};

