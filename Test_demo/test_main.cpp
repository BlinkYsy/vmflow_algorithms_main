#include <iostream>
#include <cmath>
#include <cstring>
#include "DeepLearning_AI_interface.h"
#include "DeepLearning_OCR_interface.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/utils.h>
#include <torch/script.h>
#define NOMINMAX
#include <windows.h>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <regex>
#include <numeric>
#include <string>
#include <filesystem>
#include <functional>

namespace test_AI_common {
    std::string normalizeXmlEncoding(const char* xmlText);
    std::string buildXmlResultString(const char* xmlOut);

    std::wstring decodeMultiByteString(const char* text, UINT codePage) {
        if (!text || text[0] == '\0') {
            return L"";
        }

        DWORD flags = 0;
        if (codePage == CP_UTF8) {
            flags = MB_ERR_INVALID_CHARS;
        }

        const int wideLen = MultiByteToWideChar(codePage, flags, text, -1, nullptr, 0);
        if (wideLen <= 0) {
            return L"";
        }

        std::vector<wchar_t> wideBuf(static_cast<size_t>(wideLen));
        if (MultiByteToWideChar(codePage, flags, text, -1, wideBuf.data(), wideLen) <= 0) {
            return L"";
        }

        return std::wstring(wideBuf.data());
    }

    std::string encodeWideString(const std::wstring& wideText, UINT codePage) {
        if (wideText.empty()) {
            return "";
        }

        const int byteLen = WideCharToMultiByte(
            codePage,
            0,
            wideText.c_str(),
            -1,
            nullptr,
            0,
            nullptr,
            nullptr);
        if (byteLen <= 0) {
            return "";
        }

        std::vector<char> byteBuf(static_cast<size_t>(byteLen));
        if (WideCharToMultiByte(
            codePage,
            0,
            wideText.c_str(),
            -1,
            byteBuf.data(),
            byteLen,
            nullptr,
            nullptr) <= 0) {
            return "";
        }

        return std::string(byteBuf.data());
    }

    std::string normalizeXmlEncoding(const char* xmlText) {
        if (!xmlText) {
            return "";
        }

        std::wstring wideText = decodeMultiByteString(xmlText, CP_UTF8);
        if (wideText.empty()) {
            wideText = decodeMultiByteString(xmlText, CP_ACP);
        }

        std::string utf8Text = encodeWideString(wideText, CP_UTF8);
        if (utf8Text.empty()) {
            return xmlText;
        }

        return utf8Text;
    }

    void printLocalToConsole(const std::string& text, bool isError = false);


    std::string trim(const std::string& str) {
        const char* whitespace = " \t\r\n";
        const size_t start = str.find_first_not_of(whitespace);
        if (start == std::string::npos) {
            return "";
        }
        const size_t end = str.find_last_not_of(whitespace);
        return str.substr(start, end - start + 1);
    }

    std::string extractTagValue(const std::string& xml, const std::string& tagName) {
        const std::string openTag = "<" + tagName;
        size_t start = xml.find(openTag);
        if (start == std::string::npos) {
            return "";
        }

        start = xml.find('>', start);
        if (start == std::string::npos) {
            return "";
        }

        const size_t end = xml.find("</" + tagName + ">", start);
        if (end == std::string::npos) {
            return "";
        }

        return xml.substr(start + 1, end - start - 1);
    }

    void writeUtf8XmlToFile(const std::string& xml, const std::string& outputPath) {
        std::ofstream out(outputPath, std::ios::binary);
        if (!out) {
            printLocalToConsole("Failed to open " + outputPath + " for writing.", true);
            return;
        }

        static constexpr unsigned char utf8Bom[] = { 0xEF, 0xBB, 0xBF };
        out.write(reinterpret_cast<const char*>(utf8Bom), static_cast<std::streamsize>(sizeof(utf8Bom)));
        out.write(xml.data(), static_cast<std::streamsize>(xml.size()));
    }

    void printUtf8ToConsole(const char* utf8Text) {
        if (!utf8Text) {
            return;
        }

        const std::string normalizedText = normalizeXmlEncoding(utf8Text);
        std::wstring wideText = decodeMultiByteString(normalizedText.c_str(), CP_UTF8);
        if (wideText.empty()) {
            wideText = decodeMultiByteString(utf8Text, CP_ACP);
        }

        if (wideText.empty()) {
            std::cout << utf8Text << std::endl;
            return;
        }

        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (hConsole != INVALID_HANDLE_VALUE &&
            GetConsoleMode(hConsole, &mode) &&
            WriteConsoleW(hConsole, wideText.c_str(), static_cast<DWORD>(wideText.size()), nullptr, nullptr)) {
            std::wcout << std::endl;
            return;
        }

        const std::string localText = encodeWideString(wideText, CP_ACP);
        if (!localText.empty()) {
            std::cout << localText << std::endl;
            return;
        }

        std::cout << normalizedText << std::endl;
    }

    void printUtf8ToConsole(const std::string& utf8Text) {
        printUtf8ToConsole(utf8Text.c_str());
    }

    void printLocalToConsole(const std::string& text, bool isError) {
        const std::wstring wideText = decodeMultiByteString(text.c_str(), CP_ACP);
        HANDLE hConsole = GetStdHandle(isError ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (!wideText.empty() &&
            hConsole != INVALID_HANDLE_VALUE &&
            GetConsoleMode(hConsole, &mode) &&
            WriteConsoleW(hConsole, wideText.c_str(), static_cast<DWORD>(wideText.size()), nullptr, nullptr)) {
            WriteConsoleW(hConsole, L"\n", 1, nullptr, nullptr);
            return;
        }

        if (isError) {
            std::cerr << text << std::endl;
            return;
        }

        std::cout << text << std::endl;
    }
    char* createXmlBuffer(size_t bufferSize) {
        char* xmlBuffer = new char[bufferSize];
        if (!xmlBuffer) {
            printLocalToConsole("Failed to allocate memory for xmlOut!", true);
            return nullptr;
        }

        xmlBuffer[0] = '\0';
        return xmlBuffer;
    }

    std::string buildXmlResultString(const char* xmlOut) {
        return normalizeXmlEncoding(xmlOut);
    }

    bool fileExists(const std::string& path) {
        if (path.empty()) {
            return false;
        }

        std::error_code ec;
        return std::filesystem::exists(std::filesystem::u8path(path), ec);
    }

    // 同时打印 XML 结果、归一化编码并保存到指定文件。
    std::string logAndSaveXmlResult(const char* xmlOut, const std::string& outputPath) {
        const std::string xmlResult = buildXmlResultString(xmlOut);
        printUtf8ToConsole(xmlResult);
        writeUtf8XmlToFile(xmlResult, outputPath);
        return xmlResult;
    }

    void releaseDllResources(void*& outImg, char*& xmlOut) {
        if (outImg) {
            delete static_cast<cv::Mat*>(outImg);
            outImg = nullptr;
        }

        delete[] xmlOut;
        xmlOut = nullptr;
    }

    std::string extractNamesJsonObject(const std::string& configText) {
        const std::string key = "\"names\"";
        const size_t keyPos = configText.find(key);
        if (keyPos == std::string::npos) {
            return "";
        }

        const size_t colonPos = configText.find(':', keyPos + key.length());
        if (colonPos == std::string::npos) {
            return "";
        }

        const size_t objectStart = configText.find('{', colonPos + 1);
        if (objectStart == std::string::npos) {
            return "";
        }

        int depth = 0;
        bool inString = false;
        bool escaped = false;
        for (size_t i = objectStart; i < configText.size(); ++i) {
            const char ch = configText[i];

            if (inString) {
                if (escaped) {
                    escaped = false;
                    continue;
                }

                if (ch == '\\') {
                    escaped = true;
                    continue;
                }

                if (ch == '"') {
                    inString = false;
                }
                continue;
            }

            if (ch == '"') {
                inString = true;
                continue;
            }

            if (ch == '{') {
                ++depth;
            }
            else if (ch == '}') {
                --depth;
                if (depth == 0) {
                    return configText.substr(objectStart, i - objectStart + 1);
                }
            }
        }

        return "";
    }

    // 解析 TorchScript config.txt 中的 names 映射，返回按类别编号排列的标签列表。
    std::vector<std::string> parseCategoryNamesFromConfigText(const std::string& configText) {
        const std::string namesObject = extractNamesJsonObject(configText);
        if (namesObject.empty()) {
            return {};
        }

        const std::regex pairPattern("\"(\\d+)\"\\s*:\\s*\"([^\"]*)\"");
        std::vector<std::pair<size_t, std::string>> indexedNames;
        for (std::sregex_iterator it(namesObject.begin(), namesObject.end(), pairPattern), end; it != end; ++it) {
            const size_t index = static_cast<size_t>(std::strtoul((*it)[1].str().c_str(), nullptr, 10));
            indexedNames.emplace_back(index, (*it)[2].str());
        }

        if (indexedNames.empty()) {
            return {};
        }

        size_t maxIndex = 0;
        for (const auto& item : indexedNames) {
            maxIndex = std::max(maxIndex, item.first);
        }

        std::vector<std::string> classNames(maxIndex + 1);
        for (const auto& item : indexedNames) {
            classNames[item.first] = item.second;
        }

        return classNames;
    }

    std::vector<std::string> loadCategoryNamesFromTorchScript(const std::string& modulePath) {
        if (modulePath.empty()) {
            return {};
        }

        try {
            torch::jit::ExtraFilesMap extraFiles;
            extraFiles["config.txt"] = "";
            torch::jit::load(modulePath, c10::nullopt, extraFiles);

            const auto it = extraFiles.find("config.txt");
            if (it == extraFiles.end() || it->second.empty()) {
                printLocalToConsole("No config.txt metadata found in torchscript: " + modulePath, true);
                return {};
            }

            auto classNames = parseCategoryNamesFromConfigText(it->second);
            if (classNames.empty()) {
                printLocalToConsole("No names metadata found in config.txt: " + modulePath, true);
            }
            return classNames;
        }
        catch (const c10::Error& e) {
            printLocalToConsole(std::string("Failed to load torchscript metadata: ") + e.what(), true);
        }
        catch (const std::exception& e) {
            printLocalToConsole(std::string("Failed to parse torchscript metadata: ") + e.what(), true);
        }

        return {};
    }

    // ONNX 不携带 TorchScript extra_files 元数据，这里直接跳过类别标签自动提取。
    std::vector<std::string> loadCategoryNamesForModel(const std::string& modulePath) {
        const size_t extPos = modulePath.find_last_of('.');
        const std::string ext = (extPos == std::string::npos) ? "" : modulePath.substr(extPos);
        if (_stricmp(ext.c_str(), ".onnx") == 0) {
            std::cout << "ONNX model detected. Skipping TorchScript metadata parsing." << std::endl;
        }

        return {};
    }

    // 为类别 ID 分配一个稳定的 BGR 颜色，用于可视化绘制。
    // 同一类别始终返回相同颜色，最多支持 10 种颜色循环复用。
    // 参数 category：类别 ID，负值返回灰色。
    // 返回值：对应的 cv::Scalar(B,G,R)。
    cv::Scalar getColorForCategory(int category) {
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

        if (category < 0) {
            return cv::Scalar(200, 200, 200);
        }

        return palette[static_cast<size_t>(category) % palette.size()];
    }

    std::string getCategoryDisplayName(int category, const std::vector<std::string>& classNames) {
        if (category >= 0 && static_cast<size_t>(category) < classNames.size()) {
            return classNames[static_cast<size_t>(category)];
        }

        if (category < 0) {
            return "cls_unknown";
        }

        return "cls:" + std::to_string(category);
    }

}

namespace test_AI_detect {

    namespace {
        using test_AI_common::buildXmlResultString;
        using test_AI_common::createXmlBuffer;
        using test_AI_common::extractTagValue;
        using test_AI_common::printUtf8ToConsole;
        using test_AI_common::printLocalToConsole;
        using test_AI_common::releaseDllResources;
        using test_AI_common::trim;
        using test_AI_common::writeUtf8XmlToFile;
        using test_AI_common::logAndSaveXmlResult;
        using test_AI_common::loadCategoryNamesForModel;
        using test_AI_common::loadCategoryNamesFromTorchScript;
        using test_AI_common::getColorForCategory;
        using test_AI_common::getCategoryDisplayName;

        constexpr size_t XML_BUFFER_SIZE = 1 << 16;

        std::vector<cv::RotatedRect> parseRenderRectBox(const std::string& value) {
            std::vector<cv::RotatedRect> boxes;
            std::stringstream ss(value);
            std::string entry;

            while (std::getline(ss, entry, ';')) {
                entry = trim(entry);
                if (entry.empty()) {
                    continue;
                }

                std::stringstream entryStream(entry);
                std::string token;
                std::vector<float> vals;

                while (std::getline(entryStream, token, ',')) {
                    token = trim(token);
                    if (!token.empty()) {
                        vals.push_back(std::strtof(token.c_str(), nullptr));
                    }
                }

                if (vals.size() >= 5) {
                    boxes.emplace_back(
                        cv::Point2f(vals[0], vals[1]),
                        cv::Size2f(vals[2], vals[3]),
                        vals[4]);
                }
            }

            return boxes;
        }

        // 解析 category 字段，得到每个检测框对应的类别编号。
        std::vector<int> parseCategoryList(const std::string& value) {
            std::vector<int> categories;
            std::stringstream ss(value);
            std::string token;

            while (std::getline(ss, token, ',')) {
                token = trim(token);
                if (!token.empty()) {
                    categories.push_back(std::strtol(token.c_str(), nullptr, 10));
                }
            }

            return categories;
        }

        void handleInitResult(int retCode, const char* xmlOut) {
            if (retCode == 0) {
                printLocalToConsole("Detect initialization successful!");
                logAndSaveXmlResult(xmlOut, "detect_init_result.xml");
                return;
            }

            printLocalToConsole("Detect initialization failed with error code: " + std::to_string(retCode), true);
        }

        void drawDetectDetections(
            const cv::Mat& src,
            const std::vector<cv::RotatedRect>& boxes,
            const std::vector<int>& categories,
            const std::vector<std::string>& classNames,
            const std::string& outputPath) {
            if (src.empty() || boxes.empty()) {
                return;
            }

            cv::Mat vis = src.clone();
            const bool hasCategoryPerBox = categories.size() == boxes.size();
            for (size_t boxIndex = 0; boxIndex < boxes.size(); ++boxIndex) {
                const auto& rect = boxes[boxIndex];
                const int category = hasCategoryPerBox ? categories[boxIndex] : -1;
                const cv::Scalar color = getColorForCategory(category);

                cv::Point2f pts[4];
                rect.points(pts);
                std::vector<cv::Point> contour;
                contour.reserve(4);
                for (const auto& pt : pts) {
                    contour.emplace_back(cvRound(pt.x), cvRound(pt.y));
                }
                cv::polylines(vis, contour, true, color, 2, cv::LINE_AA);

                const cv::Point2f* anchor = &pts[0];
                for (int i = 1; i < 4; ++i) {
                    if (pts[i].y < anchor->y || (pts[i].y == anchor->y && pts[i].x < anchor->x)) {
                        anchor = &pts[i];
                    }
                }

                const int textY = (static_cast<int>(anchor->y) - 8 < 20)
                    ? 20
                    : (static_cast<int>(anchor->y) - 8);
                cv::putText(
                    vis,
                    getCategoryDisplayName(category, classNames),
                    cv::Point(static_cast<int>(anchor->x), textY),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    1,
                    cv::LINE_AA);

            }

            cv::imwrite(outputPath, vis);
        }

        void saveDetectVisualizationFromXml(
            const cv::Mat& inputImage,
            const std::string& xmlResult,
            const std::vector<std::string>& classNames) {
            const std::string renderStr = extractTagValue(xmlResult, "renderRectBox");
            const std::string categoryStr = extractTagValue(xmlResult, "category");

            const auto boxes = parseRenderRectBox(renderStr);
            const auto categories = parseCategoryList(categoryStr);

            if (boxes.empty()) {
                printLocalToConsole("No renderRectBox found in XML or no detect detections.");
                return;
            }

            const std::string outputPath = "detect_result.jpg";
            drawDetectDetections(inputImage, boxes, categories, classNames, outputPath);
            printLocalToConsole("Saved detect visualization to " + outputPath + " (" + std::to_string(boxes.size()) + " boxes)");
        }

        void handleDetectResult(
            const cv::Mat& inputImage,
            void* outImg,
            const char* xmlOut,
            const std::vector<std::string>& classNames) {
            printLocalToConsole("Detect inference successful!");

            const std::string xmlResult = logAndSaveXmlResult(xmlOut, "detect_result.xml");
            saveDetectVisualizationFromXml(inputImage, xmlResult, classNames);

            cv::Mat* detectOutPtr = static_cast<cv::Mat*>(outImg);
            if (detectOutPtr && !detectOutPtr->empty()) {
                const std::string dllOutputPath = "detect_outImg.jpg";
                cv::imwrite(dllOutputPath, *detectOutPtr);
                printLocalToConsole("Saved DLL detect outImg to " + dllOutputPath);
            }
        }

        void printDetectErrorMessage(int retCode) {
            printLocalToConsole("Detect inference failed with error code: " + std::to_string(retCode), true);

            switch (retCode) {
            case -2:
                printLocalToConsole("Standard exception occurred", true);
                break;
            case -3:
                printLocalToConsole("OpenCV exception occurred", true);
                break;
            case -4:
                printLocalToConsole("Model not found", true);
                break;
            default:
                printLocalToConsole("Unknown error", true);
                break;
            }
        }

    }

    // 检测模型 测试入口：初始化模型、读取测试图、执行推理并保存结果、最后销毁模型和释放dll资源。
    int runTestDllDetect() {
        // 设置控制台代码页，保证中文 XML 或日志可以按 UTF-8 输出。
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);

        //D:\Visual Studio 2022\weights\quexian_detect\AOI20260228_s.torchscript
        //D:\Visual Studio 2022\weights\quexian_detect\AOI20260228_s.onnx
        char xmlConfig[] = R"(
        <Param>
            <DLE_AI_detect_init>
                <modulePath>..\_model\detect.onnx</modulePath>
                <moduleID>50</moduleID>
                <deviceType>1</deviceType>
                <gpuID>0</gpuID>
                <inputSize>1280</inputSize>
            </DLE_AI_detect_init>
        </Param>
    )";
        const std::string modulePath = trim(extractTagValue(xmlConfig, "modulePath"));
        //既可Onnx推理又可torchscript推理
         const std::vector<std::string> classNames = loadCategoryNamesForModel(modulePath);
        //只能Torchscript推理 -> 映射出真实类名
        //const std::vector<std::string> classNames = loadCategoryNamesFromTorchScript(modulePath);
        if (!classNames.empty()) {
            std::cout << "Loaded " << classNames.size() << " category names from model metadata." << std::endl;
        }

        char* xmlOut = createXmlBuffer(XML_BUFFER_SIZE);
        if (!xmlOut) {
            return -1;
        }

        // 调用初始化接口，加载检测模型并绑定 moduleID。
        const int ret = DLE_AI_detect_init(xmlConfig, &xmlOut);
        handleInitResult(ret, xmlOut);
        if (ret != 0) {
            std::cerr << "Detect initialization failed. Please verify the model path, model suffix, "
                << "and the compiled runtime support for TorchScript or ONNX." << std::endl;
            delete[] xmlOut;
            return ret;
        }

        // 读取用于检测的测试图片。
        cv::Mat inputImage = cv::imread("D:/Visual Studio 2022/deep-learning/0-DLL/cpm-deepai-v2.0-v1.0.0.5/test_image/output_20260317-145938.345.jpg");
        if (inputImage.empty()) {
            std::cerr << "Error: Could not load detect test image: " << std::endl;
            delete[] xmlOut;
            return -1;
        }

        void* outImg = nullptr;
        // 推理配置：moduleID 对应初始化模型，threshold 为检测阈值。
        char xmlConfig2[] = R"(
        <Param>
            <DLE_AI_detect>
                <moduleID>50</moduleID>
                <rotRectROI></rotRectROI>
                <confThreshold>0.25</confThreshold>
                <debugProcess>1</debugProcess>
                <debugResult>1</debugResult>
            </DLE_AI_detect>
        </Param>
    )";

        xmlOut[0] = '\0';

        // 调用通用检测接口执行检测推理，结果通过 xmlOut 返回。
        const int ret2 = DLE_AI_detect(&inputImage, outImg, xmlConfig2, &xmlOut);
        std::cout << "After AI_Detect, return code: " << ret2 << std::endl;

        // 成功时保存 XML 和可视化结果，失败时打印错误原因。
        if (ret2 == 0) {
            handleDetectResult(inputImage, outImg, xmlOut, classNames);
        }
        else {
            printDetectErrorMessage(ret2);
        }

        auto cleanupDetectModule = [&](char*& xmlBuffer) {
            char xmlConfig3[] = R"(
        <Param>
            <DLE_AI_detect_clean>
                <moduleID>50</moduleID>
            </DLE_AI_detect_clean>
        </Param>
    )";

            xmlBuffer[0] = '\0';
            const int cleanRet = DLE_AI_detect_clean(xmlConfig3, &xmlBuffer);
            printLocalToConsole("After DLE_AI_detect_clean, return code: " + std::to_string(cleanRet));
            if (cleanRet == 0) {
                logAndSaveXmlResult(xmlBuffer, "detect_clean_result.xml");
            }
            };

        cleanupDetectModule(xmlOut);
        // 统一释放测试过程中申请的资源。
        releaseDllResources(outImg, xmlOut);
        return 0;
    }
}

namespace test_AI_obb {

    namespace {
        using test_AI_common::buildXmlResultString;
        using test_AI_common::createXmlBuffer;
        using test_AI_common::extractTagValue;
        using test_AI_common::printUtf8ToConsole;
        using test_AI_common::releaseDllResources;
        using test_AI_common::printLocalToConsole;
        using test_AI_common::trim;
        using test_AI_common::writeUtf8XmlToFile;
        using test_AI_common::logAndSaveXmlResult;
        using test_AI_common::loadCategoryNamesForModel;
        using test_AI_common::loadCategoryNamesFromTorchScript;
        using test_AI_common::getColorForCategory;
        using test_AI_common::getCategoryDisplayName;

        // 解析 renderRectBox 字段，还原为旋转矩形列表。
        std::vector<cv::RotatedRect> parseRenderRectBox(const std::string& value) {
            std::vector<cv::RotatedRect> boxes;
            std::stringstream ss(value);
            std::string entry;

            while (std::getline(ss, entry, ';')) {
                entry = trim(entry);
                if (entry.empty()) continue;

                std::stringstream entryStream(entry);
                std::string token;
                std::vector<float> vals;

                while (std::getline(entryStream, token, ',')) {
                    token = trim(token);
                    if (!token.empty()) {
                        vals.push_back(std::strtof(token.c_str(), nullptr));
                    }
                }

                // 一个旋转框至少需要中心点、宽高和角度 5 个参数。
                if (vals.size() >= 5) {
                    cv::RotatedRect rect;
                    rect.center = cv::Point2f(vals[0], vals[1]);
                    rect.size = cv::Size2f(vals[2], vals[3]);
                    rect.angle = vals[4];
                    boxes.emplace_back(rect);
                }
            }

            return boxes;
        }

        // 解析 category 字段，将逗号分隔的类别 ID 文本转为整数列表。
        std::vector<int> parseCategoryList(const std::string& value) {
            std::vector<int> categories;
            std::stringstream ss(value);
            std::string token;

            while (std::getline(ss, token, ',')) {
                token = trim(token);
                if (!token.empty()) {
                    categories.push_back(std::strtol(token.c_str(), nullptr, 10));
                }
            }

            return categories;
        }

        // 在原图上绘制 OBB 检测框与类别标签，保存为图片。
        void drawObbDetections(
            const cv::Mat& src,
            const std::vector<cv::RotatedRect>& boxes,
            const std::vector<int>& categories,
            const std::vector<std::string>& classNames,
            const std::string& outputPath) {
            if (src.empty() || boxes.empty()) return;

            cv::Mat vis = src.clone();
            const bool hasCategoryPerBox = categories.size() == boxes.size();

            for (size_t boxIndex = 0; boxIndex < boxes.size(); ++boxIndex) {
                const auto& rect = boxes[boxIndex];
                const int category = hasCategoryPerBox ? categories[boxIndex] : -1;
                const cv::Scalar color = getColorForCategory(category);
                cv::Point2f pts[4];
                rect.points(pts);

                // 先把浮点角点转成像素坐标，再一次性用抗锯齿折线画完整个旋转框，
                // 这样会比四条独立直线看起来更顺一些。
                std::vector<cv::Point> contour;
                contour.reserve(4);
                for (const auto& pt : pts) {
                    contour.emplace_back(cvRound(pt.x), cvRound(pt.y));
                }
                cv::polylines(vis, contour, true, color, 2, cv::LINE_AA);

                if (hasCategoryPerBox) {
                    // 取最靠上的角点作为文字锚点，尽量减少标签遮挡目标主体。
                    const cv::Point2f* anchor = &pts[0];
                    for (int i = 1; i < 4; ++i) {
                        if (pts[i].y < anchor->y || (pts[i].y == anchor->y && pts[i].x < anchor->x)) {
                            anchor = &pts[i];
                        }
                    }

                    // 如果文字位置太靠近图像上边缘，则给一个最小安全边距。
                    const int textY = (static_cast<int>(anchor->y) - 8 < 20)
                        ? 20
                        : (static_cast<int>(anchor->y) - 8);
                    cv::Point textOrg(
                        static_cast<int>(anchor->x),
                        textY);

                    const std::string label = getCategoryDisplayName(category, classNames);
                    cv::putText(
                        vis,
                        label,
                        textOrg,
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv::LINE_AA);
                }
            }
            cv::imwrite(outputPath, vis);
        }

        // 统一处理模型初始化阶段的返回结果。
        void handleInitResult(int retCode, const char* xmlOut) {
            if (retCode == 0) {
                std::cout << "OBB initialization successful!" << std::endl;
                logAndSaveXmlResult(xmlOut, "obb_init_result.xml");
                return;
            }

            std::cerr << "OBB initialization failed with error code: " << retCode << std::endl;
        }

        // 从 XML 结果中提取检测框信息并生成可视化图片。
        void saveObbVisualizationFromXml(
            const cv::Mat& inputImage,
            const std::string& xmlResult,
            const std::vector<std::string>& classNames) {
            std::string renderStr = extractTagValue(xmlResult, "renderRectBox");
            std::string categoryStr = extractTagValue(xmlResult, "category");
            auto boxes = parseRenderRectBox(renderStr);
            auto categories = parseCategoryList(categoryStr);

            if (boxes.empty()) {
                std::cout << "No renderRectBox found in XML or no detections." << std::endl;
                return;
            }

            const std::string outputPath = "obb_result.jpg";
            drawObbDetections(inputImage, boxes, categories, classNames, outputPath);
            std::cout << "Saved visualization to " << outputPath << " (" << boxes.size() << " boxes)" << std::endl;
        }

        // 统一处理 OBB 推理成功后的日志、落盘与可视化流程。
        void handleObbResult(
            const cv::Mat& inputImage,
            void* outImg,
            const char* xmlOut,
            const std::vector<std::string>& classNames) {
            std::cout << "Obb successful!" << std::endl;

            const std::string xmlResult = logAndSaveXmlResult(xmlOut, "obb_result.xml");
            saveObbVisualizationFromXml(inputImage, xmlResult, classNames);

            cv::Mat* obbOutPtr = static_cast<cv::Mat*>(outImg);
            if (obbOutPtr && !obbOutPtr->empty()) {
                const std::string dllOutputPath = "obb_outImg.jpg";
                cv::imwrite(dllOutputPath, *obbOutPtr);
                printLocalToConsole("Saved DLL obb outImg to " + dllOutputPath);
            }
        }

        // 集中输出 OBB 推理失败时的错误信息。
        // DLL 约定错误码说明：
        //   -2 — C++ 标准异常
        //   -3 — OpenCV 相关异常
        //   -4 — 指定 moduleID 的模型未找到
        void printObbErrorMessage(int retCode) {
            std::cerr << "Obb inference failed with error code: " << retCode << std::endl;

            switch (retCode) {
            case -2:
                std::cerr << "Standard exception occurred" << std::endl;
                break;
            case -3:
                std::cerr << "OpenCV exception occurred" << std::endl;
                break;
            case -4:
                std::cerr << "Model not found" << std::endl;
                break;
            default:
                std::cerr << "Unknown error" << std::endl;
                break;
            }
        }
    }

    // OBB DLL 测试入口：初始化模型，执行推理，保存 XML 和可视化结果。
    int runTestDllObb() {
        // 设置控制台为 UTF-8，避免日志和 XML 中文乱码。
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);

        // 初始化 OBB 模型。
        char xmlConfig[] = R"(
        <Param>
            <DLE_AI_obb_init>
                <modulePath>..\_model\obb.torchscript</modulePath>
                <moduleID>10</moduleID>
                <deviceType>1</deviceType>
                <gpuID>0</gpuID>
                <inputSize>1280</inputSize>
            </DLE_AI_obb_init>
        </Param>
    )";
        const std::string modulePath = trim(extractTagValue(xmlConfig, "modulePath"));

        const std::vector<std::string> classNames = loadCategoryNamesForModel(modulePath);
        //映射真实类名
        //const std::vector<std::string> classNames = loadCategoryNamesFromTorchScript(modulePath);
        if (!classNames.empty()) {
            std::cout << "Loaded " << classNames.size() << " category names from torchscript metadata." << std::endl;
        }

        constexpr size_t XML_BUFFER_SIZE = 1 << 16;
        char* xmlOut = createXmlBuffer(XML_BUFFER_SIZE);
        if (!xmlOut) {
            return -1;
        }

        int ret = DLE_AI_obb_init(xmlConfig, &xmlOut);
        handleInitResult(ret, xmlOut);

        // 读取测试图像并执行推理。
        cv::Mat inputImage = cv::imread("D:/Visual Studio 2022/deep-learning/0-DLL/cpm-deepai-v2.0-v1.0.0.5/test_image/p (109).bmp");
        if (inputImage.empty()) {
            std::cerr << "Error: Could not load image!" << std::endl;
            delete[] xmlOut;
            return -1;
        }

        void* outImg = nullptr;

        // 推理参数：按 moduleID 调用已初始化的 OBB 模型。
        char xmlConfig2[] = R"(
        <Param>
            <DLE_AI_obb>
                <moduleID>10</moduleID>
                <confThreshold>0.25</confThreshold>
                <rotRectROI></rotRectROI>
                <debugProcess>1</debugProcess>
                <debugResult>1</debugResult>
            </DLE_AI_obb>
        </Param>
    )";

        xmlOut[0] = '\0';
        int ret2 = DLE_AI_obb(&inputImage, outImg, xmlConfig2, &xmlOut);
        std::cout << "After AI_Obb, return code: " << ret2 << std::endl;

        // 保存结果 XML，并在成功时输出可视化图片。
        if (ret2 == 0) {
            handleObbResult(inputImage, outImg, xmlOut, classNames);
        }
        else {
            printObbErrorMessage(ret2);
        }

        auto cleanupObbModule = [&](char*& xmlBuffer) {
            char xmlConfig3[] = R"(
        <Param>
            <DLE_AI_obb_clean>
                <moduleID>10</moduleID>
            </DLE_AI_obb_clean>
        </Param>
    )";

            xmlBuffer[0] = '\0';
            const int cleanRet = DLE_AI_obb_clean(xmlConfig3, &xmlBuffer);
            printLocalToConsole("After DLE_AI_obb_clean, return code: " + std::to_string(cleanRet));
            if (cleanRet == 0) {
                logAndSaveXmlResult(xmlBuffer, "obb_clean_result.xml");
            }
            };

        cleanupObbModule(xmlOut);
        // 释放本地资源。
        releaseDllResources(outImg, xmlOut);
        return 0;
    }
}

namespace test_AI_segment {
    namespace {
        using test_AI_common::buildXmlResultString;
        using test_AI_common::createXmlBuffer;
        using test_AI_common::extractTagValue;
        using test_AI_common::printUtf8ToConsole;
        using test_AI_common::printLocalToConsole;
        using test_AI_common::releaseDllResources;
        using test_AI_common::trim;
        using test_AI_common::writeUtf8XmlToFile;
        using test_AI_common::logAndSaveXmlResult;
        using test_AI_common::loadCategoryNamesForModel;
        using test_AI_common::loadCategoryNamesFromTorchScript;
        using test_AI_common::getCategoryDisplayName;
        using test_AI_common::getColorForCategory;


        constexpr const char* SEG_MODEL_PATH = "..\\_model\\seg_best.torchscript";
        constexpr const char* TEST_IMAGE_PATH = "D:\\Visual Studio 2022\\deep-learning\\0-DLL\\cpm-deepai-v2.0-v1.0.0.5\\test_image\\1896.rf.e718dd26abc98c2645ea37a808039034.jpg";
        constexpr const char* SEG_MASK_PATH = "segment_mask.png";
        constexpr const char* SEG_MASK_OVERLAY_PATH = "segment_mask_overlay.jpg";
        constexpr const char* SEG_RESULT_PATH = "segment_result.jpg";
        constexpr const char* SEG_INIT_XML_PATH = "segment_init_result.xml";
        constexpr const char* SEG_RESULT_XML_PATH = "segment_result.xml";
        constexpr const char* SEG_CLEAN_XML_PATH = "segment_clean_result.xml";
        constexpr size_t XML_BUFFER_SIZE = 1 << 16;

        struct InstanceSeg {
            int cls = -1;
            float conf = 0.0f;
            cv::Rect box;
            std::vector<cv::Point2f> polygon;
        };

        std::vector<cv::RotatedRect> parseRenderRectBox(const std::string& value) {
            std::vector<cv::RotatedRect> boxes;
            std::stringstream ss(value);
            std::string entry;

            while (std::getline(ss, entry, ';')) {
                entry = trim(entry);
                if (entry.empty()) continue;

                std::stringstream entryStream(entry);
                std::string token;
                std::vector<float> vals;
                while (std::getline(entryStream, token, ',')) {
                    token = trim(token);
                    if (!token.empty()) {
                        vals.push_back(std::strtof(token.c_str(), nullptr));
                    }
                }

                if (vals.size() >= 5) {
                    boxes.emplace_back(cv::Point2f(vals[0], vals[1]), cv::Size2f(vals[2], vals[3]), vals[4]);
                }
            }

            return boxes;
        }

        std::vector<int> parseIntList(const std::string& value) {
            std::vector<int> values;
            std::stringstream ss(value);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token = trim(token);
                if (!token.empty()) {
                    values.push_back(std::strtol(token.c_str(), nullptr, 10));
                }
            }
            return values;
        }

        std::vector<float> parseFloatList(const std::string& value) {
            std::vector<float> values;
            std::stringstream ss(value);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token = trim(token);
                if (!token.empty()) {
                    values.push_back(std::strtof(token.c_str(), nullptr));
                }
            }
            return values;
        }

        std::vector<cv::Point2f> parseRenderDot(const std::string& value) {
            std::vector<cv::Point2f> points;
            std::stringstream ss(value);
            std::string entry;

            while (std::getline(ss, entry, ';')) {
                entry = trim(entry);
                if (entry.empty()) continue;

                std::stringstream entryStream(entry);
                std::string xToken;
                std::string yToken;
                if (!std::getline(entryStream, xToken, ',')) continue;
                if (!std::getline(entryStream, yToken, ',')) continue;

                xToken = trim(xToken);
                yToken = trim(yToken);
                if (xToken.empty() || yToken.empty()) continue;

                points.emplace_back(
                    cvRound(std::strtof(xToken.c_str(), nullptr)),
                    cvRound(std::strtof(yToken.c_str(), nullptr)));
            }

            return points;
        }

        float pointToExpandedRectDistance(const cv::Point2f& pt, const cv::RotatedRect& rect, float margin = 3.0f) {
            const float theta = rect.angle * static_cast<float>(CV_PI / 180.0);
            const float dx = static_cast<float>(pt.x) - rect.center.x;
            const float dy = static_cast<float>(pt.y) - rect.center.y;

            const float localX = dx * std::cos(theta) + dy * std::sin(theta);
            const float localY = -dx * std::sin(theta) + dy * std::cos(theta);
            const float halfW = rect.size.width * 0.5f + margin;
            const float halfH = rect.size.height * 0.5f + margin;
            const float outX = std::abs(localX) - halfW;
            const float outY = std::abs(localY) - halfH;

            if (outX <= 0.0f && outY <= 0.0f) {
                return -std::min(-outX, -outY);
            }

            return std::hypot(std::max(0.0f, outX), std::max(0.0f, outY));
        }

        //把一整串点集，按“最接近哪个框”分组，最后得到“每个框对应一组轮廓点集”
        std::vector<std::vector<cv::Point2f>> splitRenderDotByRect(
            const std::vector<cv::Point2f>& points,
            const std::vector<cv::RotatedRect>& boxes) {

            if (points.empty()) return {};
            //如果有点但没有框，就把所有点当成一组返回
            if (boxes.empty()) return { points };

            //先按框数量创建分组容器
            std::vector<std::vector<cv::Point2f>> groups(boxes.size());
            for (const auto& pt : points) {
                size_t bestIdx = 0;
                float bestMetric = pointToExpandedRectDistance(pt, boxes[0]);
                for (size_t i = 1; i < boxes.size(); ++i) {
                    const float metric = pointToExpandedRectDistance(pt, boxes[i]);
                    if (metric < bestMetric) {
                        bestMetric = metric;
                        bestIdx = i;
                    }
                }
                groups[bestIdx].push_back(pt);
            }

            return groups;
        }

        std::vector<InstanceSeg> buildDllInstancesFromXml(const std::string& xml, const cv::Size& imageSize) {
            if (xml.empty() || imageSize.width <= 0 || imageSize.height <= 0) {
                return {};
            }

            const auto boxes = parseRenderRectBox(extractTagValue(xml, "renderRectBox"));
            const auto points = parseRenderDot(extractTagValue(xml, "renderDot"));
            //把总点集按每个框拆成若干组，每组对应一个实例的轮廓
            //必须能正确把点集按框拆分，否则类别、分数、轮廓可能会错位
            const auto groups = splitRenderDotByRect(points, boxes);
            const auto categories = parseIntList(extractTagValue(xml, "category"));
            const auto scores = parseFloatList(extractTagValue(xml, "score"));
            if (groups.empty()) {
                return {};
            }

            std::vector<InstanceSeg> instances;
            //预留容量，减少扩容开销 （提前占好“车位”，让后面 emplace_back 少折腾内存）
            instances.reserve(groups.size());
            //遍历每个轮廓组
            for (size_t i = 0; i < groups.size(); ++i) {
                const auto& group = groups[i];
                //少于 3 个点不能构成有效多边形，所以跳过
                if (group.size() < 3) continue;
                //填充单个轮廓组实例
                InstanceSeg inst;
                //groups[i]、categories[i]、scores[i] 默认认为索引是一一对应的
                inst.cls = i < categories.size() ? categories[i] : 0;
                inst.conf = i < scores.size() ? scores[i] : 0.0f;
                //保存当前实例的轮廓点
                inst.polygon = group;
                //根据轮廓点自动算一个外接矩形框
                inst.box = cv::boundingRect(group);
                //把当前实例加入返回数组
                instances.emplace_back(std::move(inst));
            }

            return instances;
        }

        bool hasDrawablePolygon(const std::vector<cv::Point2f>& polygon) {
            if (polygon.size() < 3) {
                return false;
            }

            for (const auto& pt : polygon) {
                if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) {
                    return false;
                }
            }

            return true;
        }

        std::vector<cv::Point> toContourPoints(const std::vector<cv::Point2f>& polygon) {
            std::vector<cv::Point> contour;
            contour.reserve(polygon.size());
            for (const auto& pt : polygon) {
                contour.emplace_back(cvRound(pt.x), cvRound(pt.y));
            }
            return contour;
        }

        cv::Mat buildDllMaskFromXml(const std::string& xml, const cv::Size& imageSize) {
            const auto instances = buildDllInstancesFromXml(xml, imageSize);
            if (instances.empty()) return {};

            cv::Mat mask(imageSize, CV_8U, cv::Scalar(0));
            for (const auto& inst : instances) {
                if (!hasDrawablePolygon(inst.polygon)) {
                    continue;
                }

                int gray = 0;
                switch (inst.cls) {
                case 0: gray = 64; break;
                case 1: gray = 128; break;
                case 2: gray = 192; break;
                default: gray = 255; break;
                }

                std::vector<std::vector<cv::Point>> polys = { toContourPoints(inst.polygon) };
                cv::fillPoly(mask, polys, cv::Scalar(gray), cv::LINE_8);
            }

            return mask;
        }

        cv::Mat renderOverlay(const cv::Mat& image, const std::vector<InstanceSeg>& instances, const std::vector<std::string>& classNames) {
            cv::Mat overlay = image.clone();
            for (const auto& inst : instances) {
                if (!hasDrawablePolygon(inst.polygon)) {
                    continue;
                }
                const cv::Scalar color = getColorForCategory(inst.cls);
                cv::Mat layer = overlay.clone();
                std::vector<std::vector<cv::Point>> contours = { toContourPoints(inst.polygon) };
                cv::drawContours(layer, contours, 0, color, cv::FILLED, cv::LINE_AA);
                cv::addWeighted(overlay, 0.70, layer, 0.30, 0.0, overlay);
                cv::drawContours(overlay, contours, 0, color, 2, cv::LINE_AA);

                const std::string label = getCategoryDisplayName(inst.cls, classNames);
                const int textY = (inst.box.y - 8 < 20) ? 20 : (inst.box.y - 8);
                cv::putText(
                    overlay,
                    label,
                    cv::Point(std::max(0, inst.box.x), textY),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv::LINE_AA);
            }
            return overlay;
        }

        void handleInitResult(int retCode, const char* xmlOut) {
            if (retCode == 0) {
                printLocalToConsole("Segment initialization successful!");
                logAndSaveXmlResult(xmlOut, SEG_INIT_XML_PATH);
                return;
            }

            printLocalToConsole("Segment initialization failed with error code: " + std::to_string(retCode), true);
        }

        void handleSegmentResult(const cv::Mat& inputImage, void* outImg, const char* xmlOut, const std::vector<std::string>& classNames) {
            printLocalToConsole("Segment inference successful!");
            const std::string xmlResult = logAndSaveXmlResult(xmlOut, SEG_RESULT_XML_PATH);
            const std::vector<InstanceSeg> instances = buildDllInstancesFromXml(xmlResult, inputImage.size());
            if (instances.empty()) {
                printLocalToConsole("No valid segment instances were reconstructed from XML.");
            }
            else {
                const cv::Mat overlay = renderOverlay(inputImage, instances, classNames);
                cv::imwrite(SEG_RESULT_PATH, overlay);
                printLocalToConsole("Saved segment overlay to " + std::string(SEG_RESULT_PATH) + " (" + std::to_string(instances.size()) + " instances)");
                const auto skipped = std::count_if(
                    instances.begin(),
                    instances.end(),
                    [](const InstanceSeg& inst) { return !hasDrawablePolygon(inst.polygon); });
                if (skipped > 0) {
                    printLocalToConsole("Skipped " + std::to_string(skipped) + " invalid segment polygons while rendering.");
                }
            }

            cv::Mat pureMask = buildDllMaskFromXml(xmlResult, inputImage.size());
            if (!pureMask.empty()) {
                cv::imwrite(SEG_MASK_PATH, pureMask);
                printLocalToConsole("Saved segment mask to " + std::string(SEG_MASK_PATH));
            }

            cv::Mat* segmentOutPtr = static_cast<cv::Mat*>(outImg);
            if (segmentOutPtr && !segmentOutPtr->empty()) {
                const std::string dllOutputPath = "segment_outImg.jpg";
                cv::imwrite(dllOutputPath, *segmentOutPtr);
                printLocalToConsole("Saved DLL segment outImg to " + dllOutputPath);
            }
        }

        void printSegmentErrorMessage(int retCode) {
            printLocalToConsole("Segment inference failed with error code: " + std::to_string(retCode), true);

            switch (retCode) {
            case -2:
                printLocalToConsole("Standard exception occurred", true);
                break;
            case -3:
                printLocalToConsole("OpenCV exception occurred", true);
                break;
            case -4:
                printLocalToConsole("Model not found", true);
                break;
            default:
                printLocalToConsole("Unknown error", true);
                break;
            }
        }
    }

    int runTestDllSegment() {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);

        char xmlConfig[] = R"(
        <Param>
            <DLE_AI_segment_init>
                <modulePath>..\_model\seg.torchscript</modulePath>
                <moduleID>20</moduleID>
                <deviceType>1</deviceType>
                <gpuID>0</gpuID>
                <inputSize>640</inputSize>
            </DLE_AI_segment_init>
        </Param>
    )";
        const std::string modulePath = trim(extractTagValue(xmlConfig, "modulePath"));

        const std::vector<std::string> classNames = loadCategoryNamesForModel(modulePath);
        //const std::vector<std::string> classNames = loadCategoryNamesFromTorchScript(modulePath);

        if (!classNames.empty()) {
            std::cout << "Loaded " << classNames.size() << " category names from model metadata." << std::endl;
        }

        char* xmlOut = createXmlBuffer(XML_BUFFER_SIZE);
        if (!xmlOut) {
            return -1;
        }

        const int ret = DLE_AI_segment_init(xmlConfig, &xmlOut);
        handleInitResult(ret, xmlOut);
        if (ret != 0) {
            std::cerr << "Detect initialization failed. Please verify the model path, model suffix, "
                << "and the compiled runtime support for TorchScript or ONNX." << std::endl;
            delete[] xmlOut;
            return ret;
        }

        cv::Mat inputImage = cv::imread(TEST_IMAGE_PATH);
        if (inputImage.empty()) {
            //printLocalToConsole("Could not load image: " + std::string(TEST_IMAGE_PATH), true);
            std::cerr << "Error: Could not load image!" << std::endl;
            delete[] xmlOut;
            return -1;
        }

        void* outImg = nullptr;
        char xmlConfig2[] = R"(
        <Param>
            <DLE_AI_segment>
                <moduleID>20</moduleID>
                <confThreshold>0.25</confThreshold>
                <rotRectROI></rotRectROI>
                <debugProcess>1</debugProcess>
                <debugResult>1</debugResult>
            </DLE_AI_segment>
        </Param>
    )";

        xmlOut[0] = '\0';
        const int ret2 = DLE_AI_segment(&inputImage, outImg, xmlConfig2, &xmlOut);
        std::cout << "After DLE_AI_segment, return code: " << ret2 << std::endl;

        if (ret2 == 0) {
            handleSegmentResult(inputImage, outImg, xmlOut, classNames);
        }
        else {
            printSegmentErrorMessage(ret2);
        }

        auto cleanupSegmentModule = [&](char*& xmlBuffer) {
            char xmlConfig3[] = R"(
        <Param>
            <DLE_AI_segment_clean>
                <moduleID>20</moduleID>
            </DLE_AI_segment_clean>
        </Param>
    )";

            xmlBuffer[0] = '\0';
            const int cleanRet = DLE_AI_segment_clean(xmlConfig3, &xmlBuffer);
            printLocalToConsole("After DLE_AI_segment_clean, return code: " + std::to_string(cleanRet));
            if (cleanRet == 0) {
                logAndSaveXmlResult(xmlBuffer, SEG_CLEAN_XML_PATH);
            }
        };

        cleanupSegmentModule(xmlOut);
        releaseDllResources(outImg, xmlOut);
        return 0;
    }
}

namespace test_AI_pose {

    namespace {
        using test_AI_common::buildXmlResultString;
        using test_AI_common::createXmlBuffer;
        using test_AI_common::extractTagValue;
        using test_AI_common::printUtf8ToConsole;
        using test_AI_common::releaseDllResources;
        using test_AI_common::trim;
        using test_AI_common::writeUtf8XmlToFile;
        using test_AI_common::printLocalToConsole;
        using test_AI_common::logAndSaveXmlResult;
        using test_AI_common::loadCategoryNamesForModel;
        using test_AI_common::loadCategoryNamesFromTorchScript;
        using test_AI_common::getColorForCategory;
        using test_AI_common::getCategoryDisplayName;

        // DLL XML 输出接口使用的缓冲区大小。
        constexpr size_t XML_BUFFER_SIZE = 1 << 16;

        // 解析 renderRectBox 字段，将分号分隔的旋转框转换为 OpenCV 结构。
        std::vector<cv::RotatedRect> parseRenderRectBox(const std::string& value) {
            std::vector<cv::RotatedRect> boxes;
            std::stringstream ss(value);
            std::string entry;

            while (std::getline(ss, entry, ';')) {
                entry = trim(entry);
                if (entry.empty()) {
                    continue;
                }

                std::stringstream entryStream(entry);
                std::string token;
                std::vector<float> vals;
                while (std::getline(entryStream, token, ',')) {
                    token = trim(token);
                    if (!token.empty()) {
                        vals.push_back(std::strtof(token.c_str(), nullptr));
                    }
                }

                if (vals.size() >= 5) {
                    boxes.emplace_back(
                        cv::Point2f(vals[0], vals[1]),
                        cv::Size2f(vals[2], vals[3]),
                        vals[4]);
                }
            }

            return boxes;
        }

        // 解析 category 字段，得到每个检测框对应的类别编号。
        std::vector<int> parseCategoryList(const std::string& value) {
            std::vector<int> categories;
            std::stringstream ss(value);
            std::string token;

            while (std::getline(ss, token, ',')) {
                token = trim(token);
                if (!token.empty()) {
                    categories.push_back(std::strtol(token.c_str(), nullptr, 10));
                }
            }

            return categories;
        }

        // 解析 renderDot 字段，将关键点列表转换为浮点坐标集合。
        std::vector<cv::Point2f> parseRenderDots(const std::string& value) {
            std::vector<cv::Point2f> points;
            std::stringstream ss(value);
            std::string entry;

            while (std::getline(ss, entry, ';')) {
                entry = trim(entry);
                if (entry.empty()) {
                    continue;
                }

                std::stringstream entryStream(entry);
                std::string token;
                std::vector<float> vals;
                while (std::getline(entryStream, token, ',')) {
                    token = trim(token);
                    if (!token.empty()) {
                        vals.push_back(std::strtof(token.c_str(), nullptr));
                    }
                }

                if (vals.size() >= 2) {
                    points.emplace_back(vals[0], vals[1]);
                }
            }

            return points;
        }

        // 处理 pose 模块初始化结果，并在成功时保存初始化 XML。
        void handleInitResult(int retCode, const char* xmlOut) {
            if (retCode == 0) {
                std::cout << "Pose initialization successful!" << std::endl;
                logAndSaveXmlResult(xmlOut, "pose_init_result.xml");
                return;
            }

            std::cerr << "Pose initialization failed with error code: " << retCode << std::endl;
        }

        // 将旋转框、类别标签和关键点绘制到原图副本上，并保存可视化结果。
        void drawPoseDetections(
            const cv::Mat& src,
            const std::vector<cv::RotatedRect>& boxes,
            const std::vector<int>& categories,
            const std::vector<std::string>& classNames,
            const std::vector<cv::Point2f>& keypoints,
            int keypointsPerBox,
            const std::string& outputPath) {
            if (src.empty() || boxes.empty()) {
                return;
            }

            cv::Mat vis = src.clone();
            const bool hasCategoryPerBox = categories.size() == boxes.size();

            for (size_t boxIndex = 0; boxIndex < boxes.size(); ++boxIndex) {
                const auto& rect = boxes[boxIndex];
                const int category = hasCategoryPerBox ? categories[boxIndex] : -1;
                const cv::Scalar color = getColorForCategory(category);

                // 将旋转框四个顶点转换为整数坐标并绘制闭合轮廓。
                cv::Point2f pts[4];
                rect.points(pts);
                std::vector<cv::Point> contour;
                contour.reserve(4);
                for (const auto& pt : pts) {
                    contour.emplace_back(cvRound(pt.x), cvRound(pt.y));
                }
                cv::polylines(vis, contour, true, color, 2, cv::LINE_AA);

                // 选择最靠上的顶点作为标签锚点，避免文字遮挡框中心区域。
                const cv::Point2f* anchor = &pts[0];
                for (int i = 1; i < 4; ++i) {
                    if (pts[i].y < anchor->y || (pts[i].y == anchor->y && pts[i].x < anchor->x)) {
                        anchor = &pts[i];
                    }
                }

                const int textY = (static_cast<int>(anchor->y) - 8 < 20)
                    ? 20
                    : (static_cast<int>(anchor->y) - 8);
                cv::putText(
                    vis,
                    getCategoryDisplayName(category, classNames),
                    cv::Point(static_cast<int>(anchor->x), textY),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv::LINE_AA);

                // 按每个目标对应的关键点数量逐个绘制点，并用折线连接。
                if (keypointsPerBox > 0) {
                    const size_t startIndex = boxIndex * static_cast<size_t>(keypointsPerBox);
                    const size_t endIndex = std::min(keypoints.size(), startIndex + static_cast<size_t>(keypointsPerBox));
                    cv::Point prevPoint;
                    bool hasPrevPoint = false;

                    for (size_t kpIndex = startIndex; kpIndex < endIndex; ++kpIndex) {
                        const cv::Point center(
                            cvRound(keypoints[kpIndex].x),
                            cvRound(keypoints[kpIndex].y));

                        cv::circle(vis, center, 4, color, -1, cv::LINE_AA);
                        if (hasPrevPoint) {
                            cv::line(vis, prevPoint, center, color, 2, cv::LINE_AA);
                        }

                        prevPoint = center;
                        hasPrevPoint = true;
                    }
                }
            }

            cv::imwrite(outputPath, vis);
        }

        // 从 DLL 返回的 XML 中解析 pose 结果，并生成可视化图像。
        void savePoseVisualizationFromXml(
            const cv::Mat& inputImage,
            const std::string& xmlResult,
            const std::vector<std::string>& classNames) {
            const std::string renderStr = extractTagValue(xmlResult, "renderRectBox");
            const std::string categoryStr = extractTagValue(xmlResult, "category");
            const std::string renderDotStr = extractTagValue(xmlResult, "renderDot");

            const auto boxes = parseRenderRectBox(renderStr);
            const auto categories = parseCategoryList(categoryStr);
            const auto keypoints = parseRenderDots(renderDotStr);

            if (boxes.empty()) {
                std::cout << "No renderRectBox found in XML or no pose detections." << std::endl;
                return;
            }

            int keypointsPerBox = 0;
            if (!keypoints.empty() && keypoints.size() % boxes.size() == 0) {
                keypointsPerBox = static_cast<int>(keypoints.size() / boxes.size());
            }

            const std::string outputPath = "pose_result.jpg";
            drawPoseDetections(inputImage, boxes, categories, classNames, keypoints, keypointsPerBox, outputPath);
            std::cout << "Saved pose visualization to " << outputPath
                << " (" << boxes.size() << " boxes, " << keypoints.size() << " keypoints)" << std::endl;
        }

        // 处理推理成功结果，保存 XML 并生成 pose 可视化图片。
        void handlePoseResult(
            const cv::Mat& inputImage,
            void* outImg,
            const char* xmlOut,
            const std::vector<std::string>& classNames) {
            std::cout << "Pose inference successful!" << std::endl;

            const std::string xmlResult = logAndSaveXmlResult(xmlOut, "pose_result.xml");
            savePoseVisualizationFromXml(inputImage, xmlResult, classNames);

            cv::Mat* poseOutPtr = static_cast<cv::Mat*>(outImg);
            if (poseOutPtr && !poseOutPtr->empty()) {
                const std::string dllOutputPath = "pose_outImg.jpg";
                cv::imwrite(dllOutputPath, *poseOutPtr);
                printLocalToConsole("Saved DLL pose outImg to " + dllOutputPath);
            }
        }

        // 根据返回码打印常见错误原因，便于快速定位问题。
        void printPoseErrorMessage(int retCode) {
            std::cerr << "Pose inference failed with error code: " << retCode << std::endl;

            switch (retCode) {
            case -2:
                std::cerr << "Standard exception occurred" << std::endl;
                break;
            case -3:
                std::cerr << "OpenCV exception occurred" << std::endl;
                break;
            case -4:
                std::cerr << "Model not found" << std::endl;
                break;
            default:
                std::cerr << "Unknown error" << std::endl;
                break;
            }
        }

    }

    // pose 测试入口：初始化 DLL、执行推理、保存 XML 和可视化结果。
    int runTestDllPose() {
        // 将控制台输入输出编码设置为 UTF-8，确保中文 XML 和日志显示正常。
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);

        char xmlConfig[] = R"(
        <Param>
            <DLE_AI_pose_init>
                <modulePath>..\_model\pose_kpt.torchscript</modulePath>
                <moduleID>30</moduleID>
                <deviceType>1</deviceType>
                <gpuID>0</gpuID>
                <inputSize>1280</inputSize>
            </DLE_AI_pose_init>
        </Param>
    )";

        // 先从初始化 XML 中取出模型路径，再尝试读取 torchscript 自带的类别名。
        const std::string modulePath = trim(extractTagValue(xmlConfig, "modulePath"));

        //const std::vector<std::string> classNames = loadCategoryNamesFromTorchScript(modulePath);
        const std::vector<std::string> classNames = loadCategoryNamesForModel(modulePath);
        if (!classNames.empty()) {
            std::cout << "Loaded " << classNames.size() << " category names from torchscript metadata." << std::endl;
        }

        char* xmlOut = createXmlBuffer(XML_BUFFER_SIZE);
        if (!xmlOut) {
            return -1;
        }

        // 第一步：初始化 pose 模块。
        const int ret = DLE_AI_pose_init(xmlConfig, &xmlOut);
        handleInitResult(ret, xmlOut);
        if (ret != 0) {
            delete[] xmlOut;
            return ret;
        }

        cv::Mat inputImage = cv::imread(
            "D:/Visual Studio 2022/deep-learning/0-DLL/cpm-deepai-v2.0-v1.0.0.5/test_image/p (109).bmp");
        if (inputImage.empty()) {
            std::cerr << "Error: Could not load pose test image!" << std::endl;
            delete[] xmlOut;
            return -1;
        }

        void* outImg = nullptr;
        char xmlConfig2[] = R"(
        <Param>
            <DLE_AI_pose>
                <moduleID>30</moduleID>
                <confThreshold>0.5</confThreshold>
                <rotRectROI></rotRectROI>
                <debugProcess>1</debugProcess>
                <debugResult>1</debugResult>
            </DLE_AI_pose>
        </Param>
    )";

        xmlOut[0] = '\0';

        // 第二步：执行 pose 推理，并拿到 XML 结果和输出图像。
        const int ret2 = DLE_AI_pose(&inputImage, outImg, xmlConfig2, &xmlOut);
        std::cout << "After AI_Pose, return code: " << ret2 << std::endl;

        if (ret2 == 0) {
            handlePoseResult(inputImage, outImg, xmlOut, classNames);
        }
        else {
            printPoseErrorMessage(ret2);
        }

        auto cleanupPoseModule = [&](char*& xmlBuffer) {
            char xmlConfig3[] = R"(
        <Param>
            <DLE_AI_pose_clean>
                <moduleID>30</moduleID>
            </DLE_AI_pose_clean>
        </Param>
    )";

            xmlBuffer[0] = '\0';
            const int cleanRet = DLE_AI_pose_clean(xmlConfig3, &xmlBuffer);
            printLocalToConsole("After DLE_AI_pose_clean, return code: " + std::to_string(cleanRet));
            if (cleanRet == 0) {
                logAndSaveXmlResult(xmlBuffer, "pose_clean_result.xml");
            }
            };
        //销毁AI_pose模块
        cleanupPoseModule(xmlOut);
        releaseDllResources(outImg, xmlOut);
        return 0;
    }
}

namespace test_AI_ocr {
    namespace {
        using test_AI_common::buildXmlResultString;
        using test_AI_common::createXmlBuffer;
        using test_AI_common::extractTagValue;
        using test_AI_common::fileExists;
        using test_AI_common::printUtf8ToConsole;
        using test_AI_common::releaseDllResources;
        using test_AI_common::writeUtf8XmlToFile;
        using test_AI_common::printLocalToConsole;
        using test_AI_common::logAndSaveXmlResult;

        constexpr size_t XML_BUFFER_SIZE = 1 << 16;
        constexpr const char* PRIMARY_ONNX_MODEL_PATH =
            "..\\_model\\en_PP-OCRv4_rec_infer.onnx";
        constexpr const char* PRIMARY_DICT_PATH =
            "..\\_model\\en_dict.txt";
        constexpr const char* OCR_TEST_IMAGE_PATH = "C:\\Users\\yusiy\\Desktop\\456.png";
            //"D:\\Visual Studio 2022\\deep-learning\\0-DLL\\cpm-deepai-v2.0-v1.0.0.5\\test_image\\ocr\\11_2.png";

        // 解析 OCR 返回 XML 里的 recRegion 字段，恢复文本区域框。
        std::vector<cv::RotatedRect> parseRenderRectBox(const std::string& value) {
            std::vector<cv::RotatedRect> boxes;
            size_t begin = 0;
            while (begin < value.size()) {
                const size_t end = value.find(';', begin);
                const std::string item = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
                begin = end == std::string::npos ? value.size() : end + 1;
                if (item.empty()) {
                    continue;
                }

                std::vector<float> vals;
                size_t tokenBegin = 0;
                while (tokenBegin < item.size()) {
                    const size_t tokenEnd = item.find(',', tokenBegin);
                    const std::string token = item.substr(tokenBegin, tokenEnd == std::string::npos ? std::string::npos : tokenEnd - tokenBegin);
                    tokenBegin = tokenEnd == std::string::npos ? item.size() : tokenEnd + 1;
                    if (!token.empty()) {
                        vals.push_back(std::strtof(token.c_str(), nullptr));
                    }
                }

                if (vals.size() >= 5) {
                    boxes.emplace_back(
                        cv::Point2f(vals[0], vals[1]),
                        cv::Size2f(vals[2], vals[3]),
                        vals[4]);
                }
            }
            return boxes;
        }

        // 处理 OCR 初始化结果，成功时保存初始化返回 XML。
        void handleInitResult(int retCode, const char* xmlOut) {
            if (retCode == 0) {
                std::cout << "OCR initialization successful!" << std::endl;
                logAndSaveXmlResult(xmlOut, "ocr_init_result.xml");
                return;
            }
            std::cerr << "OCR initialization failed with error code: " << retCode << std::endl;
        }

        // 在原图上绘制 OCR 文本区域，并输出简单可视化图片。
        void drawOcrRegion(const cv::Mat& src, const std::vector<cv::RotatedRect>& boxes, const std::string& text) {
            if (src.empty() || boxes.empty()) {
                return;
            }

            cv::Mat vis = src.clone();
            for (const auto& rect : boxes) {
                cv::Point2f pts[4];
                rect.points(pts);
                std::vector<cv::Point> contour;
                contour.reserve(4);
                for (const auto& pt : pts) {
                    contour.emplace_back(cvRound(pt.x), cvRound(pt.y));
                }
                cv::polylines(vis, contour, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::putText(
                    vis,
                    text.empty() ? "" : text,
                    contour.front(),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA);
            }

            cv::imwrite("ocr_result.jpg", vis);
        }

        // 处理 OCR 推理结果：保存 XML、打印识别文本，并在有框时导出可视化图。
        void handleOcrResult(const cv::Mat& inputImage, void* outImg, const char* xmlOut) {
            const std::string xmlResult = logAndSaveXmlResult(xmlOut, "ocr_result.xml");
            const std::string text = extractTagValue(xmlResult, "recText");
            const std::string score = extractTagValue(xmlResult, "score");
            const auto boxes = parseRenderRectBox(extractTagValue(xmlResult, "renderRectBox"));

            std::cout << "OCR inference successful!" << std::endl;
            std::cout << "Recognized text: " << text << std::endl;
            std::cout << "Score: " << score << std::endl;

            if (!boxes.empty()) {
                drawOcrRegion(inputImage, boxes, text);
            }

            cv::Mat* ocrOutPtr = static_cast<cv::Mat*>(outImg);
            if (ocrOutPtr && !ocrOutPtr->empty()) {
                const std::string dllOutputPath = "ocr_outImg.jpg";
                cv::imwrite(dllOutputPath, *ocrOutPtr);
                printLocalToConsole("Saved DLL ocr outImg to " + dllOutputPath);
            }
        }

        // 根据 DLL 返回码输出更易读的 OCR 错误信息。
        void printOcrErrorMessage(int retCode) {
            std::cerr << "OCR inference failed with error code: " << retCode << std::endl;
            switch (retCode) {
            case -2:
                std::cerr << "Standard exception occurred" << std::endl;
                break;
            case -3:
                std::cerr << "OpenCV exception occurred" << std::endl;
                break;
            case -4:
                std::cerr << "Model not found" << std::endl;
                break;
            default:
                std::cerr << "Unknown error" << std::endl;
                break;
            }
        }

    }

    // OCR ONNX 测试入口：初始化模型，执行识别，并保存 XML 与结果图。
    int runTestDllOcr() {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);

        // 检查模型和字典文件是否存在。
        const std::string modulePath = PRIMARY_ONNX_MODEL_PATH;
        const std::string dictPath = PRIMARY_DICT_PATH;
        if (!fileExists(modulePath)) {
            std::cerr << "ONNX model file does not exist: " << modulePath << std::endl;
            return -1;
        }
        if (!fileExists(dictPath)) {
            std::cerr << "OCR dictionary file does not exist: " << PRIMARY_DICT_PATH << std::endl;
            return -1;
        }

        // 组装 OCR 初始化参数。
        //std::string xmlConfigStr =
        //    "<Param>"
        //    "    <DLE_ocr_init>"
        //    "        <modulePath>" + modulePath + "</modulePath>"
        //    "        <characterDictPath>" + dictPath + "</characterDictPath>"
        //    "        <moduleID>70</moduleID>"
        //    "        <moduleType>4</moduleType>"
        //    "        <deviceType>1</deviceType>"
        //    "        <gpuID>0</gpuID>"
        //    "        <inputHeight>48</inputHeight>"
        //    "        <inputSize>320</inputSize>"
        //    "        <useSpaceChar>1</useSpaceChar>"
        //    "    </DLE_ocr_init>"
        //    "</Param>";
        //std::vector<char> xmlConfigBuffer(xmlConfigStr.begin(), xmlConfigStr.end());
        //xmlConfigBuffer.push_back('\0');

        char xmlConfig[] = R"(
            <Param>
                <DLE_ocr_init>
                    <modulePath>..\_model\PP-OCRv6_inference.onnx</modulePath>
                    <characterDictPath>..\_model\PP-OCRv6_dict.txt</characterDictPath>
                    <moduleID>70</moduleID>
                    <deviceType>1</deviceType>
                    <gpuID>0</gpuID>
                    <inputHeight>48</inputHeight>
                    <inputWidth>640</inputWidth>
                    <useSpaceChar>0</useSpaceChar>
                </DLE_ocr_init>
            </Param>
            )";

        char* xmlOut = createXmlBuffer(XML_BUFFER_SIZE);
        if (!xmlOut) {
            return -1;
        }

        //const int ret = DLE_ocr_init(xmlConfigBuffer.data(), &xmlOut);
        const int ret = DLE_ocr_init(xmlConfig, &xmlOut);

        handleInitResult(ret, xmlOut);
        if (ret != 0) {
            delete[] xmlOut;
            return ret;
        }

        // 读取测试图片并执行 OCR 推理。
        cv::Mat inputImage = cv::imread(OCR_TEST_IMAGE_PATH);
        if (inputImage.empty()) {
            std::cerr << "Error: Could not load OCR test image: " << OCR_TEST_IMAGE_PATH << std::endl;
            delete[] xmlOut;
            return -1;
        }

        void* outImg = nullptr;
        // 推理参数需指定已初始化的 moduleID。
        char xmlConfig2[] = R"(
        <Param>
            <DLE_ocr>
                <moduleID>70</moduleID>
                <rotRectROI></rotRectROI>
                <debugProcess>1</debugProcess>
                <debugResult>1</debugResult>
                <ocrConfThreshold>0.5</ocrConfThreshold>
            </DLE_ocr>
        </Param>
    )";

        xmlOut[0] = '\0';
        const int ret2 = DLE_ocr(&inputImage, outImg, xmlConfig2, &xmlOut);
        std::cout << "After AI_OCR, return code: " << ret2 << std::endl;

        // 成功时保存结果，失败时输出错误原因。
        if (ret2 == 0) {
            handleOcrResult(inputImage, outImg, xmlOut);
        }
        else {
            printOcrErrorMessage(ret2);
        }

        auto cleanupPoseModule = [&](char*& xmlBuffer) {
            char xmlConfig3[] = R"(
        <Param>
            <DLE_ocr_clean>
                <moduleID>70</moduleID>
            </DLE_ocr_clean>
        </Param>
    )";

            xmlBuffer[0] = '\0';
            const int cleanRet = DLE_ocr_clean(xmlConfig3, &xmlBuffer);
            printLocalToConsole("After DLE_ocr_clean, return code: " + std::to_string(cleanRet));
            if (cleanRet == 0) {
                logAndSaveXmlResult(xmlBuffer, "ocr_clean_result.xml");
            }
            };
        //销毁AI_pose模块
        cleanupPoseModule(xmlOut);
        // 释放本地资源。
        releaseDllResources(outImg, xmlOut);
        return 0;
    }
}

namespace {
    void configureRuntimePaths() {
        std::vector<wchar_t> exePath(MAX_PATH);
        DWORD pathLen = GetModuleFileNameW(nullptr, exePath.data(), static_cast<DWORD>(exePath.size()));
        if (pathLen == 0 || pathLen >= exePath.size()) {
            return;
        }
        std::wstring exeDir(exePath.data(), pathLen);
        const size_t lastSlash = exeDir.find_last_of(L"\\/");
        if (lastSlash == std::wstring::npos) {
            return;
        }
        exeDir.erase(lastSlash);
        SetCurrentDirectoryW(exeDir.c_str());
        SetDllDirectoryW(exeDir.c_str());
    }

    int runSelectedCase(const std::string& selection, int argc = 0, char** argv = nullptr) {
        if (selection == "1") {
            std::cout << "Running test_AI_obb ..." << std::endl;
            return test_AI_obb::runTestDllObb();
        }
        if (selection == "2") {
            std::cout << "Running test_AI_segment ..." << std::endl;
            return test_AI_segment::runTestDllSegment();
        }
        if (selection == "3") {
            std::cout << "Running test_AI_pose ..." << std::endl;
            return test_AI_pose::runTestDllPose();
        }
        if (selection == "4") {
            std::cout << "Running test_AI_detect ..." << std::endl;
            return test_AI_detect::runTestDllDetect();
        }
        if (selection == "5") {
            std::cout << "Running test_AI_ocr ..." << std::endl;
            return test_AI_ocr::runTestDllOcr();
        }
       /* if (selection == "6") {
            std::cout << "Running test_AI_qrcode ..." << std::endl;
            return test_qrcode::main_QRcode(argc, argv);
        }*/
        std::cerr << "Invalid selection: " << selection << std::endl;
        std::cerr << "Available options: 1 / test_AI_obb, 2 / test_AI_segment, 3 / test_AI_pose, 4 / test_AI_detect, 5 / test_AI_ocr, 6 / test_qrcode" << std::endl;
        return -1;
    }
}

int main(int argc, char* argv[]) {
    //configureRuntimePaths();
    if (argc > 1) {
        return runSelectedCase(argv[1], argc - 1, argv + 1);
    }
    std::string selection;
    std::cout << "Please select which test to run:" << std::endl;
    std::cout << "  1. test_AI_obb" << std::endl;
    std::cout << "  2. test_AI_segment" << std::endl;
    std::cout << "  3. test_AI_pose" << std::endl;
    std::cout << "  4. test_AI_detect" << std::endl;
    std::cout << "  5. test_AI_ocr" << std::endl;
    std::cout << "  6. test_qrcode" << std::endl;
    std::cout << "Input 1, 2, 3, 4, 5, 6: ";
    std::getline(std::cin, selection);
    return runSelectedCase(selection);
}

//int main()
//{
//    return test_AI_detect::runTestDllDetect();
//}
