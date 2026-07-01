#pragma once
#include <Windows.h>
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

class CMarkup;

class DeepLearning_AI_pose
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
	string							m_pPoseKptScore;										// 关键点分数输出字符串
	int								m_pPoseNumKpts;											// 单个目标的关键点数量
	int								m_pPoseMetaNumKpts = 0;									// 从模型元信息里读取到的关键点数量
	bool							m_pPosePreferAngle = false;								// 表示是否优先认为模型输出里包含obb角度信息
	// 模型后端参数
	torch::jit::Module				m_pModule;												// TorchScript 模型对象与执行设备
	c10::Device						m_pDevice = c10::Device(c10::kCPU);						// LibTorch 设备类型，默认 CPU
	ModelRuntimeFormat				m_modelRuntimeFormat = ModelRuntimeFormat::TorchScript; // 当前模型实际采用的推理后端
	string							m_pModulePath;											// 模型路径与公共推理参数
	int								m_pDeviceType = 0;										// 是否使用 GPU 推理
	float							m_pThreshold = 0.25;									// 姿态关键点Pose时默认置信度阈值
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
	// 初始化或复用一个 DeepLearning_AI_pose 实例，并输出初始化结果 XML
	int initPoseAiObj(char* xmlIn, char** xmlOut, std::string apiName);
	// 从初始化 XML 中解析 moduleID
	int parseInitPoseXml(char* xmlIn, std::string apiName);
	// 执行单张图像的目标检测任务
	int processOneImgByPose(void* inImg, void*& outImg, char* xmlIn, char** xmlOut, std::string apiName);
	// 按 moduleID 销毁实例并输出销毁结果 XML
	int destoryPoseObject(char* xmlIn, char** xmlOut, std::string apiName);
	// 姿态关键点Pose检测推理，输出框、类别、分数及关键点
	int runPose(cv::Mat& m_pSrcImg, std::vector<cv::RotatedRect>& DC_VecRotaRect, std::string& DC_score, std::string& DC_class, std::vector<cv::Point2f>& DC_Pts, cv::Rect& rectSrc);
	// 输出初始化结果 XML
	int outputInitXml(char* xmlOut, std::string apiName);
	// 生成销毁接口的返回 XML
	void outputDesXml(char* xmlOut, std::string apiName);
	// 将旋转框 ROI 转成普通矩形 ROI
	void rotateRect2Rect();

private:
	// 包括张量本体、letterbox 的 padding 和缩放系数
	struct InferenceInput
	{
		torch::Tensor tensor;
		float pad_w;
		float pad_h;
		float scale;
	};

	// 姿态模型输出布局说明，用于兼容不同关键点维度配置
	struct PoseLayout
	{
		int nc;
		int num_kps;
		int kpt_dim;
		bool has_angle;
		int angle_index;
		int kpt_start;
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
	// 根据输出维度推断姿态模型布局
	bool inferPoseLayout(int feature_dim, PoseLayout& layout);
	// 姿态输出过滤、NMS 与拼装
	torch::Tensor buildPoseDetections(torch::Tensor pred, float conf_thres, float iou_thres, const PoseLayout& layout);
	// 将单个检测张量转换为原图坐标系下的旋转框
	cv::RotatedRect tensorBoxToRotatedRect(const torch::Tensor& det_row, float pad_w, float pad_h, float scale, const cv::Rect& rectSrc);
	// 将姿态检测结果写入旋转框、分数、类别和关键点输出容器
	void appendPoseDetections(const torch::Tensor& det, const PoseLayout& layout, float pad_w, float pad_h, float scale,
		const cv::Rect& rectSrc, std::vector<cv::RotatedRect>& boxes, std::string& scores,
		std::string& classes, std::vector<cv::Point2f>& points);
	// 执行 letterbox 缩放与补边，返回 padding 与缩放系数
	std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(640, 640));
	// NMS 核心实现，返回保留下来的索引集合
	at::Tensor nms_kernel(const at::Tensor& dets, const at::Tensor& scores, float iou_threshold);
	static void ensurePoseAiLoggerInitialized();
	static int getFixedSquareOnnxInputSize(const Ort::Session& session);
	static std::string trimCopy(const std::string& value);
	static int parsePoseKptCountFromShapeText(const std::string& shapeText);
	static int parsePoseKptCountFromConfigText(const std::string& configText);
	static bool parsePosePreferAngleFromConfigText(const std::string& configText);
	static bool isValidKeypoint(const cv::Point2f& pt, const cv::Size& imageSize);
	static bool isPointNearRotatedRect(const cv::Point2f& pt, const cv::RotatedRect& rect, float margin);
	static std::vector<int> parseCategoryIds(const std::string& value);
	static cv::Scalar getColorForCategory(int category);
	static std::string getCategoryDisplayName(int category);
	static cv::Mat renderPoseImage(
		const cv::Mat& srcImg,
		const std::vector<cv::RotatedRect>& rects,
		const std::vector<int>& categories,
		const std::vector<cv::Point2f>& keypoints);
	static renderParam makeRenderParam(const std::vector<cv::RotatedRect>& rects, const std::vector<cv::Point2f>& dots);
	static void beginProcessXml(CMarkup& xml, DeepLearning_AI_pose* deepAI, const std::string& apiName, const char* moduleStatusCh);
	static void finishProcessXml(CMarkup& xml, char** xmlOut);
	static void addScoreCategoryXml(CMarkup& xml, const std::string& score, const std::string& category, bool trimTrailingComma);
	static void addAxisXml(CMarkup& xml, const std::vector<cv::RotatedRect>& rects, bool useRectAngle, const char* shortAxisName, const char* longAxisName);
	static void addRenderXml(CMarkup& xml, renderParam& rendPar, const char* rectCh, bool includeDots);
	static std::string trimTrailingCommaOrDefault(const std::string& value, const char* defaultValue);
	static std::string toLowerCopy(const std::string& value);
	static torch::ScalarType ortTensorElementTypeToTorch(ONNXTensorElementDataType element_type);
	static torch::Tensor ortValueToTensor(Ort::Value& value);
	static DeepLearning_AI_pose* findDeepAIByModuleXml(char* xmlIn, const std::string& apiName);
	static void prepareProcessXmlInput(DeepLearning_AI_pose* deepAI, void* inImg, char* xmlIn, const std::string& apiName);
	static void saveDebugSourceImage(DeepLearning_AI_pose* deepAI);
	static void saveDebugResultImage(
		DeepLearning_AI_pose* deepAI,
		const std::vector<cv::RotatedRect>& rects,
		const std::vector<cv::Point2f>& dots);
	static cv::Mat cropByPreparedRectRoi(DeepLearning_AI_pose* deepAI, cv::Rect& rectSrc);
	static void writeObjectResultXml(
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
		const char* rectCh);
	static void writeNormalResultXml(
		char** xmlOut,
		DeepLearning_AI_pose* deepAI,
		const std::string& apiName,
		int defectStatus,
		bool includeRender,
		bool includeAxis,
		const char* shortAxisName,
		const char* longAxisName);
	static void logWholeProcessCost(DWORD startTick);
	static void buildAxisStrings(
		const std::vector<cv::RotatedRect>& rects,
		bool useRectAngle,
		std::string& shortAxisStr,
		std::string& longAxisStr,
		std::string& angleStr);

};
