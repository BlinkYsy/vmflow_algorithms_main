#pragma once
#include <opencv2/opencv.hpp>

using std::string;

/*
	常量
*/
//CString OUT_INT = "%d";														//输出int类型数据
//CString OUT_FLOAT = "%0.5f";												//输出5位小数的float类型数据

/*
	基本参数
*/

//圆形ROI
struct circleROI
{
	float centerX;															//圆心点X
	float centerY;															//圆心点Y
	float outerRadius;														//外环半径
	float innerRadius;														//内环半径
	float startAngle;														//起始角度
	float angleRange;														//角度范围
};

//直线
struct linePoints
{
	cv::Point2f linePt1;
	cv::Point2f linePt2;
};

/*
	图像二值化
*/
enum binaryTypeEnum
{
	binaryType_HardThreshold = 0,											//硬阈值二值化
	binaryType_MeanThreshold = 1,											//均值二值化
	binaryType_GaussianThreshold = 2,										//高斯二值化
	binaryType_OTSU = 3													    //自动二值化
};

enum operation
{
	greaterEqual = 0,                                                       //大于等于
	lessEqual = 1,                                                          //小于等于
	equalTo = 2,                                                            //等于
	notQqual = 3                                                            //不等于
};

/*
	形态学处理
*/
enum morphTypeEnum
{
	morphType_Dilate = 0,													//膨胀
	morphType_Erode = 1,													//腐蚀
	morphType_Open = 2,														//开运算
	morphType_Close = 3														//闭运算
};

enum morphShapeEnum
{
	morphShape_Rectange = 0,												//矩形
	morphShape_Ellipse = 1,													//椭圆
	morphShape_Crosss = 2													//十字
};

/*
	blob工具
*/

//阈值方式
enum thresholdTypeEnum
{
	THRESH_BINARYIMAGE = 0,													//不进行二值化
	THRESH_SINGLETHRESH = 1,												//单阈值
	THRESH_DOUBLETHRESH = 2,												//双阈值
	THRESH_AUTOTHRESH = 3,													//自动阈值
	THRESH_FIXEDSOFTTHRESH = 4,												//软阈值（固定）
	THRESH_RELATIVESOFTTHRESH = 5											//软阈值（相对）
};

enum contourTypeEnum
{
	contourType_RETR_EXTERNAL = 0,														//外轮廓
	contourType_RETR_LIST = 1,															//全部轮廓
	contourType_RETR_TREE = 2,															//全部继承轮廓
	contourType_RETR_CCOMP = 3															//两级继承轮廓
};

//极性
enum polarityEnum
{
	POLARITY_DARKONBRIGHT = 0,												//亮于背景
	POLARITY_BRIGHTONDARK = 1												//暗于背景
};

//排序特征
enum sortFeatureEnum
{
	SORTFEATURE_AREA = 0,													//面积
	SORTFEATURE_PERIMETER = 1,												//周长
	SORTFEATURE_CIRCULARITY = 2,											//圆形度
	SORTFEATURE_RECT = 3,													//矩形度
	SORTFEATURE_CENTROID_X = 4,												//连通域中心X
	SORTFEATURE_CENTROID_Y = 5,												//连通域中心Y
	SORTFEATURE_BOXANGLE = 6,												//Box角度
	SORTFEATURE_BOXWIDTH = 7,												//Box宽
	SORTFEATURE_BOXHEIGHT = 8,												//Box高
	SORTFEATURE_RECT_X = 9,													//矩形左上顶点X
	SORTFEATURE_RECT_Y = 10,												//矩形左上顶点Y
	SORTFEATURE_AXISANGLE = 11,												//二阶中心距主轴角度
	SORTFEATURE_AXISRATIO = 12												//轴比(box短轴/box长轴)
};

//排序方式
enum sortModeEnum
{
	SORTMODE_ASCEND = 0,													//升序
	SORTMODE_DECEND = 1,													//降序
	SORTMODE_NOTSORT = 2													//不排序
};

//连通性
enum connectivityEnum
{
	CONNECTIVITY_8 = 0,														//8连通
	CONNECTIVITY_4 = 1														//4连通
};

/*
	颜色抽取
*/

//颜色空间
enum colorSpaceEnum
{
	colorSpace_RGB = 0,														//RGB
	colorSpace_HSV = 1,														//HSV
	colorSpace_HSI = 2														//HSI
};

/*
	颜色转换
*/

//转换类型
enum colorTransformTypeEnum
{
	colorTransformType_RGB2GRAY = 0,										//RGB转灰度
	colorTransformType_RGB2HSV = 1,											//RGB转HSV
	colorTransformType_RGB2HSI = 2,											//RGB转HSI
	colorTransformType_RGB2YUV = 3											//RGB转YUV
};

//显示通道
enum showChannelEnum
{
	showChannel_FirstChannel = 0,											//第一通道
	showChannel_SecondChannel = 1,											//第二通道
	showChannel_ThirdChannel = 2											//第三通道
};

/*
	点线测量
*/

/*
	直线边缘缺陷检测
*/

enum findModeEnum
{
	findMode_Best = 0,														//最强
	findMode_First = 1,														//第一条
	findMode_Last = 2														//最后一条
};

enum edgePolarityEnum
{
	edgePolarity_B2W = 0,													//从黑到白
	edgePolarity_W2B = 1,													//从白到黑
	edgePolarity_ALL = 2													//任意极性
};

enum flawPolarityEnum
{
	FlawPolarityEnable_All = 0,												//轨迹两侧缺陷
	FlawPolarityEnable_RightBottom = 1,										//轨迹右侧缺陷
	FlawPolarityEnable_LeftTop = 2											//轨迹左侧缺陷
};

/*
	快速匹配
*/
enum matchThresholdFlagEnum
{
	matchThresholdFlag_Auto = 0,											//自动阈值
	matchThresholdFlag_Model = 1,											//模板阈值
	matchThresholdFlag_Manual = 2											//手动阈值
};

enum matchFastPolarityEnum
{
	polarity_No = 0,														//不考虑极性
	polarity_Yes = 1														//考虑极性
};

enum strategyEnum
{
	strategy_PRECISE = 0,													//精确匹配
	strategy_GENERAL = 1,													//普通匹配
	strategy_SKETCHY = 2													//粗略匹配
};

/*
	顶点检测
*/
enum peakEdgePolarity
{
	edgePolarity_Up = 0,													//从下往上
	edgePolarity_Down = 1,													//从上往下
	edgePolarity_Left = 2,													//从左往右
	edgePolarity_Right = 3													//从右往左
};
