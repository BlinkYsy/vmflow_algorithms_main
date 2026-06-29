# vmflow_algorithms_main 项目代码说明与调通指南

本文档根据当前仓库源码整理，目标是说明本项目的整体作用、目录结构、主要 DLL 接口、模型文件、构建运行方式以及常见调试入口。

## 1. 项目整体作用

`vmflow_algorithms_main` 是一个 Windows C++/MSVC 视觉算法工程集合，根解决方案为 `vmflow_algorithms_main.sln`。项目面向 DLL 形式的算法集成，使用 C 风格导出接口和 XML 参数传递，主要包含三类能力：

- `DeepLearning_AI_tool`：AI 推理模块，支持目标检测、语义分割、OBB 定向检测、姿态关键点检测Pose。
- `DeepLearning_OCR`：OCR 识别模块，基于 ONNX Runtime 加载 OCR 模型。
- `ColorProcess`：传统图像颜色处理模块，支持颜色抽取和颜色转换。

仓库还包含 `Test_demo` 测试程序，用于演示如何初始化模型、调用 DLL 接口、读取 XML 输出并进行可视化或结果保存。

## 2. 目录结构

```text
.
├── vmflow_algorithms_main.sln        根解决方案，包含 AI、OCR、颜色处理和测试 Demo 工程
├── DeepLearning_AI_tool/             AI 推理工程，输出 DeepLearning_AI_tool DLL
├── DeepLearning_OCR/                 OCR 推理工程，输出 DeepLearning_OCR DLL
├── ColorProcess/                     颜色处理工程，输出 ColorProcess DLL
├── Test_demo/                        测试程序工程，演示 DLL 调用流程
├── _common_files/                    公共工具、XML 解析、日志、几何/图像辅助函数
├── _model/                           测试和部署用模型文件、字典文件
├── _testimage/                       测试图片示例
├── docs/                             运行时依赖的 DLL、VMFlow接口文档说明等其他辅助文件
├── .gitattributes                    Git LFS 二进制文件跟踪规则
└── .gitignore                        VS 缓存、编译产物、本机用户配置忽略规则
```

## 3. Visual Studio 工程说明

| 工程 | 类型 | 作用 |
| --- | --- | --- |
| `DeepLearning_AI_tool/DeepLearning_AI_tool.vcxproj` | DynamicLibrary | AI 推理 DLL。Release x64 配置链接 OpenCV、LibTorch、ONNX Runtime。 |
| `DeepLearning_OCR/DeepLearning_OCR.vcxproj` | DynamicLibrary | OCR 推理 DLL。Release x64 配置链接 OpenCV、LibTorch、ONNX Runtime。 |
| `ColorProcess/ColorProcess.vcxproj` | DynamicLibrary | 颜色处理 DLL。主要依赖 OpenCV。 |
| `Test_demo/Test_demo.vcxproj` | Application | 测试程序。链接 AI DLL import lib、OpenCV、LibTorch 等库，演示检测、OBB、分割、姿态、OCR 调用。 |

当前解决方案包含以下配置：

- `Debug | x64`
- `Debug | x86`
- `Release | x64`
- `Release | x86`

实际调试建议优先使用 `Release | x64`，因为 AI/OCR 模型推理和第三方运行时 DLL 通常按 x64 Release 环境准备。

## 4. 主要模块职责

### 4.1 `DeepLearning_AI_tool`

AI 推理模块对外暴露 `DLE_AI_*` 系列接口。每种任务都有初始化、推理、销毁三类函数，内部通过 `moduleID` 管理已经加载的模型实例。

| 文件 | 作用 |
| --- | --- |
| `DeepLearning_AI_interface.h/.cpp` | DLL 导出接口声明和实现。负责把外部调用分发到具体任务模块。 |
| `DeepLearning_AI_detect.*` | 目标检测实现，支持 `.onnx`、`.torchscript` 模型路径。 |
| `DeepLearning_AI_segment.*` | 实例分割实现，支持模型加载、ROI 处理、mask/结果输出。 |
| `DeepLearning_AI_obb.*` | OBB 旋转目标检测实现，输出旋转框及角点信息。 |
| `DeepLearning_AI_pose.*` | 姿态识别实现，输出关键点/检测结果。 |

### 4.2 `DeepLearning_OCR`

OCR 模块对外暴露 `DLE_ocr_*` 系列接口。当前实现主要面向 `.onnx` 模型，使用 ONNX Runtime 加载 OCR 推理模型，并通过 XML 指定 `moduleID`、`modulePath` 等参数。

| 文件 | 作用 |
| --- | --- |
| `DeepLearning_OCR_interface.h/.cpp` | OCR DLL 导出接口声明和实现。 |
| `DeepLearning_OCR.cpp/.h` | OCR 模型加载、推理、XML 解析和结果输出。 |

### 4.3 `ColorProcess`

颜色处理模块提供传统图像算法接口，不依赖深度学习模型。

| 文件 | 作用 |
| --- | --- |
| `ColorProcess_interface.h/.cpp` | 颜色处理 DLL 导出接口。 |
| `ColorProcess_color_extract.*` | 颜色抽取逻辑。 |
| `ColorProcess_color_transform.*` | 颜色空间/通道转换逻辑。 |

### 4.4 `_common_files`

公共代码目录，供多个工程复用。

| 文件 | 作用 |
| --- | --- |
| `ResolveXml.*` | 基于 `CMarkup` 的 XML 参数读取工具。 |
| `Markup.*` | 内嵌 XML 解析库。 |
| `renderParam.*` | 检测框、点集、几何结果到 XML 字段的序列化辅助函数。 |
| `parseBaseParam.*` | 通用基础参数解析。 |
| `FileLogEx.*` | 文件日志工具。 |
| `Util.*` | 图像、字符串、几何等通用辅助函数。 |
| `commStu.h` | 通用结构体、枚举和参数定义。 |
| `CPM_deepAI_interface.h`、`CPM_ocr_interface.h` | 兼容/集成层接口定义。 |

## 5. DLL 对外接口

### 5.1 AI 推理接口

```cpp
int DLE_AI_detect_init(char* xmlIn, char** xmlOut);
int DLE_AI_detect(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
int DLE_AI_detect_clean(char* xmlIn, char** xmlOut);

int DLE_AI_segment_init(char* xmlIn, char** xmlOut);
int DLE_AI_segment(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
int DLE_AI_segment_clean(char* xmlIn, char** xmlOut);

int DLE_AI_obb_init(char* xmlIn, char** xmlOut);
int DLE_AI_obb(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
int DLE_AI_obb_clean(char* xmlIn, char** xmlOut);

int DLE_AI_pose_init(char* xmlIn, char** xmlOut);
int DLE_AI_pose(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
int DLE_AI_pose_clean(char* xmlIn, char** xmlOut);
```

### 5.2 OCR 接口

```cpp
int DLE_ocr_init(char* xmlIn, char** xmlOut);
int DLE_ocr(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
int DLE_ocr_clean(char* xmlIn, char** xmlOut);
```

### 5.3 颜色处理接口

```cpp
int CPR_color_extract(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
int CPR_color_transform(void* inImg, void*& outImg, char* xmlIn, char** xmlOut);
```

调用约定基本一致：

1. 初始化接口通过 XML 读取 `moduleID`、`modulePath`、设备配置等信息，并加载模型。
2. 推理/处理接口接收 `cv::Mat` 指针、XML 输入和 XML 输出缓冲区。
3. 清理接口通过 `moduleID` 释放对应模型实例。

## 6. 模型与运行时文件

`_model/` 目录保存当前测试使用的模型和字典文件：

| 文件类型 | 说明 |
| --- | --- |
| `*.onnx` | ONNX Runtime 推理模型，覆盖检测、分割、OBB、OCR 等任务。 |
| `*.torchscript` | LibTorch/TorchScript 推理模型。 |
| `*.txt` | OCR 字典、类别名或模型配套文本资源。 |

`docs/` 目录包含部分运行时 DLL：

- `opencv_world440.dll`
- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`
- `onnxruntime_providers_cuda.dll`
- `torch_cpu.dll`
- `c10.dll`
- `libiomp5md.dll`
- `uv.dll`

注意：模型和 DLL 文件体积较大，仓库通过 Git LFS 管理这些二进制文件。首次克隆后需要确认 Git LFS 已安装，并执行：

```powershell
git lfs pull
```

## 7. 构建与运行环境

推荐环境：

- Windows 10/11
- Visual Studio 2022
- MSVC v143 工具集
- Windows SDK
- OpenCV 4.4.0
- LibTorch
- ONNX Runtime
- CUDA Toolkit / NVIDIA 驱动，可选，仅 GPU 推理时需要

构建方式：

1. 用 Visual Studio 打开 `vmflow_algorithms_main.sln`。
2. 选择 `Release | x64`。
3. 先编译 DLL 工程：
   - `DeepLearning_AI_tool`
   - `DeepLearning_OCR`
   - `ColorProcess`
4. 再编译 `Test_demo`。
5. 确认运行时 DLL 位于 exe 同目录或已加入 `PATH`。

也可以在 Developer PowerShell 中使用 MSBuild：

```powershell
msbuild "E:\vmflow_algorithms_main\vmflow_algorithms_main.sln" /p:Configuration=Release /p:Platform=x64
```

如果普通 PowerShell 找不到 `msbuild`，请使用 “Developer PowerShell for VS 2022” 或把 VS/MSBuild 路径加入环境变量。

## 8. XML 调用流程

### 8.1 初始化模型

以目标检测为例：

```xml
<Param>
  <DLE_AI_detect_init>
    <modulePath>..\_model\detect.onnx</modulePath>
    <moduleID>50</moduleID>
    <deviceType>0</deviceType>
    <gpuID>0</gpuID>
    <inputSize>640</inputSize>
  </DLE_AI_detect_init>
</Param>
```

关键字段：

| 字段 | 说明 |
| --- | --- |
| `modulePath` | 模型路径。AI 模块支持 `.onnx`、`.torchscript`；OCR 模块主要使用 `.onnx`。 |
| `moduleID` | 模型实例 ID。后续推理和清理必须使用同一个 ID。 |
| `deviceType` | 设备类型。通常 `0=CPU`，`1=GPU`。 |
| `gpuID` | GPU 编号。 |
| `inputSize` | 模型输入尺寸。 |

### 8.2 执行推理

```xml
<Param>
  <DLE_AI_detect>
    <moduleID>50</moduleID>
    <threshold>50</threshold>
    <rotRectROI></rotRectROI>
    <debugProcess>0</debugProcess>
    <debugResult>0</debugResult>
  </DLE_AI_detect>
</Param>
```

通用字段：

| 字段 | 说明 |
| --- | --- |
| `moduleID` | 已初始化的模型实例 ID。 |
| `threshold` | 置信度或任务阈值，具体解释由对应模块决定。 |
| `rotRectROI` | 旋转矩形 ROI，空值通常表示整图处理。 |
| `debugProcess` | 是否输出处理中间图像或日志。 |
| `debugResult` | 是否输出调试结果。 |

### 8.3 清理模型

```xml
<Param>
  <DLE_AI_detect_clean>
    <moduleID>50</moduleID>
  </DLE_AI_detect_clean>
</Param>
```

清理接口会根据 `moduleID` 找到对应实例并释放资源。实际集成时建议保证初始化、推理、清理成对出现，避免重复加载模型或长期占用显存/内存。

## 9. Test_demo 调试入口

`Test_demo/test_main.cpp` 中已经包含多类任务的调用样例，主要演示：

- `DLE_AI_detect_init` / `DLE_AI_detect` / `DLE_AI_detect_clean`
- `DLE_AI_obb_init` / `DLE_AI_obb` / `DLE_AI_obb_clean`
- `DLE_AI_segment_init` / `DLE_AI_segment` / `DLE_AI_segment_clean`
- `DLE_AI_pose_init` / `DLE_AI_pose` / `DLE_AI_pose_clean`
- `DLE_ocr_init` / `DLE_ocr` / `DLE_ocr_clean`

样例中常用模型路径包括：

```text
..\_model\detect.onnx
..\_model\obb.torchscript
..\_model\seg.torchscript
..\_model\pose_kpt.torchscript
..\_model\PP-OCRv6_inference.onnx
```

调试时建议先确认：

1. 模型文件真实存在，并且 Git LFS 已拉取完整文件。
2. 测试图片路径存在，`cv::imread` 返回非空图像。
3. `moduleID` 在初始化、推理、清理阶段保持一致。
4. 运行目录能找到 OpenCV、LibTorch、ONNX Runtime 相关 DLL。

## 10. 常见问题与排查

| 现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| 编译时报找不到 `opencv_world440.lib` | OpenCV lib 路径未配置或路径不一致 | 检查工程属性中的附加库目录。 |
| 编译时报找不到 `torch.lib`、`torch_cpu.lib`、`c10.lib` | LibTorch 路径未配置 | 检查 LibTorch include/lib/bin 路径。 |
| 编译时报找不到 `onnxruntime.lib` | ONNX Runtime SDK 路径未配置 | 补充 ONNX Runtime include/lib 路径。 |
| 启动时报 DLL 缺失 | 运行时 DLL 不在 exe 同目录或 PATH 中 | 将 `docs/` 中 DLL 复制到 exe 目录，或配置 PATH。 |
| 初始化返回失败 | 模型路径错误、模型格式不支持、运行时版本不匹配 | 检查 `modulePath`、模型扩展名、ONNX/LibTorch 版本。 |
| 推理返回找不到模型实例 | `moduleID` 未初始化或 ID 不一致 | 先调用对应 init 接口，并保持同一个 `moduleID`。 |
| `cv::imread` 读取失败 | 图片路径不存在或格式不支持 | 改成真实图片路径，确认图像非空。 |
| XML 输出中文乱码 | 控制台编码或 XML 字符集不一致 | 使用 UTF-8 保存 XML/源码，必要时参考 `Test_demo` 中的编码转换逻辑。 |
| GPU 推理失败 | CUDA、驱动、ONNX Runtime CUDA Provider、LibTorch CUDA 版本不匹配 | 先用 CPU 路径验证，再逐项检查 CUDA 运行时依赖。 |

## 11. 最小调通 checklist

- [ ] 安装 Visual Studio 2022 和 MSVC v143 C++ 桌面开发组件。
- [ ] 准备 OpenCV 4.4.0，并确认 include/lib/bin 路径。
- [ ] 准备 LibTorch，并确认 include/lib/bin 路径。
- [ ] 准备 ONNX Runtime，并确认 include/lib/bin 路径。
- [ ] 执行 `git lfs pull`，确保 `_model/` 和 `docs/` 中大文件完整。
- [ ] 使用 `Release | x64` 编译三个 DLL 工程。
- [ ] 编译并运行 `Test_demo`。
- [ ] 确认测试图片路径、模型路径和 `moduleID`。
- [ ] 查看控制台返回码、XML 输出和调试日志。
