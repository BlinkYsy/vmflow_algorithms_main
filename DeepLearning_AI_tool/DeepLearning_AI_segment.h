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

class DeepLearning_AI_segment
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
	float							m_pThreshold = 0.25;									// 分割时默认置信度阈值
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
	// 语义分割推理，输出掩膜图和分数/类别信息
	int runSegment(cv::Mat& m_pSrcImg, cv::Mat& m_pDstImg, std::string& DC_score, std::string& DC_class);
	// 初始化或复用一个 DeepLearning_AI_segment 实例，并输出初始化结果 XML
	int initSegmentAiObj(char* xmlIn, char** xmlOut, std::string apiName);
	// 从初始化 XML 中解析 moduleID
	int parseInitSegmentXml(char* xmlIn, std::string apiName);
	// 执行单张图像的目标检测任务
	int processOneImgBySegment(void* inImg, void*& outImg, char* xmlIn, char** xmlOut, std::string apiName);
	// 按 moduleID 销毁实例并输出销毁结果 XML
	int destorySegmentObject(char* xmlIn, char** xmlOut, std::string apiName);
	// 用于输出分割图像结果
	int outputSegmentImg(cv::Mat inImg, void*& outImg);
	// 输出初始化结果 XML
	int outputInitXml(char* xmlOut, std::string apiName);
	// 生成销毁接口返回 XML
	void outputDesXml(char* xmlOut, std::string apiName);
	// 将旋转框 ROI 转成普通矩形 ROI
	void rotateRect2Rect();
	// 将 ROI 裁剪到图像边界范围内，避免越界访问
	void judgeROI(const cv::Mat imgIn);

private:
	// 统一封装 YOLO 系列模型预处理结果：
	// 包括张量tensor本体、letterbox 的 padding 和缩放系数
	struct InferenceInput
	{
		torch::Tensor tensor;
		float pad_w;
		float pad_h;
		float scale;
	};
	// 当前缓存的灰度值映射表是按照多少个类别 nc 生成的
	int m_cachedSegNc = 0;
	// 灰度值映射表的数组（用来保存每个 class_idx 对应的灰度值）
	std::vector<int> m_segClassGrayValues;

	// 根据类别ID构建灰度-idx映射表
	void buildSegClassGrayValues(int nc);
	//根据 preds 的输出维度推算分割类别数 nc，TorchScript/ONNX 的 Tensor 输出都可复用
	int inferSegNcFromPreds(const torch::Tensor& preds);
	// 从 TorchScript forward 的 IValue 输出中取出 preds，并推算分割类别数 nc
	int inferSegNcFromTorchscriptOutput(const torch::jit::IValue& output);
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
	// 记录单阶段耗时日志
	void logStageCost(const char* stage_name, std::chrono::high_resolution_clock::time_point start);
	// 分割输出的后处理入口，负责生成 mask、框、分数和类别信息
	int PostProcessing_seg(const torch::Tensor& preds, const torch::Tensor& proto, cv::Mat& m_pDstImg,
		float pad_w, float pad_h, float scale, const cv::Size& m_pSrcImg_shape,
		float conf_thres, float iou_thres, std::string& DC_score, std::string& DC_class);
	// 执行 letterbox 缩放与补边，返回 padding 与缩放系数
	std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(640, 640));
	// NMS 核心实现，返回保留下来的索引集合
	at::Tensor nms_kernel(const at::Tensor& dets, const at::Tensor& scores, float iou_threshold);
	// 将二维张量浅包装成 OpenCV Mat
	cv::Mat tensor2Mat(const torch::Tensor& input_tensor);
};
