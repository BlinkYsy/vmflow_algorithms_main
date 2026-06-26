//#ifndef __FILELOG_H__
//#define __FILELOG_H__

#include <stdio.h>
#include <atlstr.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>//用于格式化时间戳
#include <ctime>
#include <string>
#include <vector>
#include <chrono>

#ifdef _WIN32
#include <direct.h> // Windows 下的 mkdir 用于创建目录
#else
#include <sys/stat.h> // Linux 下的 mkdir 用于创建目录
#endif

using std::string;
using std::vector;

#define LOG_MESSAGE 1
#define LOG_ERROR 2
#define LOG_WARNING 3
#define LOG_INFORMATION 4
#define LOG_SGIMAGE 5

#define BOOL bool

//日志类
class Logger {
public:
    Logger(const std::string& basePath, const std::string& logName = "log")
        : basePath_(basePath), logName_(logName), currentFileSize_(0), maxFileSize_(1024 * 1024 * 1024) { // 1GB
        // 确保基础目录存在
        if (!directoryExists(basePath_)) {
            createDirectory(basePath_);
        }
    }

    ~Logger() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
    }

    void log(const std::string& message) {
        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);

        // 使用 localtime_s 替换 localtime
        std::tm now_tm;
        errno_t err = localtime_s(&now_tm, &now_time);
        if (err != 0) {
            throw std::runtime_error("Failed to convert time to local time structure");
        }

        // 按日期创建日志目录
        std::string dateDir = formatTime(now_tm, "%Y-%m-%d");
        std::string logDir = basePath_ + "/" + dateDir;
        if (!directoryExists(logDir)) {
            createDirectory(logDir);
        }

        // 检查是否需要创建新的日志文件
        std::string logFilePath = getLogFilePath(logDir, now_tm);
        if (!logFile_.is_open() || currentLogFile_ != logFilePath || currentFileSize_ >= maxFileSize_) {
            if (logFile_.is_open()) {
                logFile_.close();
            }
            logFile_.open(logFilePath, std::ios::out | std::ios::app);
            if (!logFile_.is_open()) {
                throw std::runtime_error("Failed to open log file: " + logFilePath);
            }
            currentLogFile_ = logFilePath;
            currentFileSize_ = getFileSize(logFilePath);
        }

        // 写入日志
        std::string logEntry = "[" + formatTime(now_tm, "%Y-%m-%d %H:%M:%S") + "] " + message + "\n";
        logFile_ << logEntry;
        logFile_.flush(); // 确保日志立即写入文件

        // 更新当前文件大小
        currentFileSize_ += logEntry.size();

        /*
        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_time);

        // 按日期创建日志目录
        std::string dateDir = formatTime(now_tm, "%Y-%m-%d");
        std::string logDir = basePath_ + "/" + dateDir;
        if (!directoryExists(logDir)) {
            createDirectory(logDir);
        }

        // 检查是否需要创建新的日志文件
        std::string logFilePath = getLogFilePath(logDir, now_tm);
        if (!logFile_.is_open() || currentLogFile_ != logFilePath || currentFileSize_ >= maxFileSize_) {
            if (logFile_.is_open()) {
                logFile_.close();
            }
            logFile_.open(logFilePath, std::ios::out | std::ios::app);
            if (!logFile_.is_open()) {
                throw std::runtime_error("Failed to open log file: " + logFilePath);
            }
            currentLogFile_ = logFilePath;
            currentFileSize_ = getFileSize(logFilePath);
        }

        // 写入日志
        std::string logEntry = "[" + formatTime(now_tm, "%Y-%m-%d %H:%M:%S") + "] " + message + "\n";
        logFile_ << logEntry;
        logFile_.flush(); // 确保日志立即写入文件

        // 更新当前文件大小
        currentFileSize_ += logEntry.size();
        */
    }

    void setLogName(const std::string& logName) {
        logName_ = logName;
    }

private:
    std::string basePath_;       // 基础路径
    std::string logName_;        // 日志文件名
    std::ofstream logFile_;      // 日志文件流
    std::string currentLogFile_; // 当前日志文件路径
    size_t currentFileSize_;     // 当前日志文件大小
    const size_t maxFileSize_;   // 单个日志文件最大大小（1GB）

    // 检查目录是否存在
    bool directoryExists(const std::string& path) {
        struct stat info;
        return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
    }

    // 创建目录
    void createDirectory(const std::string& path) {
#ifdef _WIN32
        _mkdir(path.c_str()); // Windows 下的 mkdir
#else
        mkdir(path.c_str(), 0777); // Linux 下的 mkdir
#endif
    }

    // 格式化时间
    std::string formatTime(const std::tm& tm, const std::string& format) {
        char buffer[80];
        strftime(buffer, sizeof(buffer), format.c_str(), &tm);
        return std::string(buffer);
    }

    // 获取日志文件路径
    std::string getLogFilePath(const std::string& logDir, const std::tm& tm) {
        std::stringstream ss;
        ss << logDir << "/" << logName_ << "_" << formatTime(tm, "%Y%m%d");

        // 如果文件已存在，检查是否需要追加序号
        size_t index = 1;
        std::string filePath;
        do {
            filePath = ss.str() + (index > 1 ? "_" + std::to_string(index) : "") + ".log";
            index++;
        } while (fileExists(filePath) && getFileSize(filePath) >= maxFileSize_);

        return filePath;
    }

    // 检查文件是否存在
    bool fileExists(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }

    // 获取文件大小
    size_t getFileSize(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return 0;
        }
        return file.tellg();
    }
};

class CFileLogEx
{
private:
	bool m_bOpened;
	long m_nMaxLogSize;
	long m_nFiles;
	char m_FileName[2048];
	FILE *m_fp;
	CComAutoCriticalSection m_cs;

	string m_logPath;
	string m_logName;

	int m_preDay;	// 日志存放到不同日期的目录
	int m_curDay;

public:
	CFileLogEx();
	~CFileLogEx();

	string getYearMouthDay();

	BOOL dataChange();
	int delPreLogDir();
	BOOL initLog(char* logPath, const char* FileName);
	BOOL Open(char* logPath, const char* FileName);
	void Close(void);
	BOOL AddMessage(const char *s1, const char *s2);
	BOOL AddLog(long LogTypeCode, /*const char *Source, */const char *Content);
	void LogSystemError(HMODULE hSource, DWORD id);
	inline BOOL IsOpened() { return m_bOpened; };
	inline void SetNumberFiles(long n) { if (n < 2) m_nFiles = 2; else m_nFiles = n; };
	inline void SetMaxLogSize(long n) { m_nMaxLogSize = n; };
	bool AddLogEx(long type, const char* msg, ...);
private:
	void Backup(long nMax);	
};

//#endif
