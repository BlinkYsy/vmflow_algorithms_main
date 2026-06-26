//#include "pch.h"
//#include "stdafx.h"
//#include "resource.h"
#include "FileLogEx.h"
//#include <share.h>
#include <time.h>
#include <string>
#include <io.h>
#include <vector>

using namespace std;

CFileLogEx::CFileLogEx()
{
	m_bOpened = false;
	m_nMaxLogSize = 50000L;
	m_fp = NULL;
	m_nFiles = 6000;
	m_FileName[0] = '\0';
}

CFileLogEx::~CFileLogEx()
{
	Close();
}

bool CFileLogEx::AddLogEx(long type, const char* msg, ...)
{

	va_list vl_list;
	va_start(vl_list, msg);
	char content[1024000] = { 0 };
	vsprintf_s(content, msg, vl_list);   //格式化处理msg到字符串
	va_end(vl_list);
	return AddLog(type, content);

}

string CFileLogEx::getYearMouthDay()
{
	SYSTEMTIME st;
	::GetLocalTime(&st);
	CString s;
	s.Format("%04d-%02d-%02d", st.wYear, st.wMonth, st.wDay);
	return s.GetBuffer();
}

int CFileLogEx::delPreLogDir()
{
	vector<string> veLogPath;

	long handle;                                                //用于查找的句柄
	struct _finddata_t fileinfo;                          //文件信息的结构体
	handle = _findfirst(m_logPath.c_str(), &fileinfo);         //第一次查找
	if (-1 == handle)
		return -1;
	//printf("%s\n", fileinfo.name);                         //打印出找到的文件的文件名
	while (!_findnext(handle, &fileinfo))               //循环查找其他符合的文件，知道找不到其他的为止
	{
		veLogPath.push_back(fileinfo.name);
		//printf("%s\n", fileinfo.name);
	}
	_findclose(handle);

}

BOOL CFileLogEx::initLog(char* logPath, const char* FileName)
{
	m_logPath = logPath;
	m_logName = FileName;

	SYSTEMTIME st;
	::GetLocalTime(&st);
	CString s;
	s.Format("%02d", st.wDay);
	m_preDay = atoi(s.GetBuffer());

	return true;
}

BOOL CFileLogEx::dataChange()
{
	SYSTEMTIME st;
	::GetLocalTime(&st);
	CString s;
	s.Format("%02d", st.wDay);
	m_curDay = atoi(s.GetBuffer());

	if (m_curDay != m_preDay)
	{
		m_preDay = m_curDay;
		return true;
	}
	else
		return false;
}

BOOL CFileLogEx::Open(char* logPath, const char* FileName)
{
	Close();
	char sName[512] = { 0 };
	//strcpy(sName, FileName);
	string sYMD = getYearMouthDay();
	sprintf_s(sName, "%s//%s//%s", logPath, (char*)sYMD.c_str(), FileName);

	// 创建文件夹
	char sPath[512] = { 0 };
	sprintf_s(sPath, "%s//%s//", logPath, (char*)sYMD.c_str());
	if (_access(sPath, 0) == -1) // 没有文件夹创建文件夹
	{
		CreateDirectoryA(sPath, 0);
		sprintf_s(sName, "%s//%s//%s", logPath, (char*)sYMD.c_str(), FileName);
	}

	if (m_FileName == NULL)
		return false;

	// 使用 fopen_s 替换 fopen
	FILE* temp_fp = NULL;
	errno_t err = fopen_s(&temp_fp, sName, "a");
	m_fp = temp_fp; // 保持原有成员变量赋值

	if (err != 0) // 文件打开失败
	{
		string sTmp = sName;
		string path, sfile;

		string::size_type len = sTmp.find('/');
		string::size_type rlen = sTmp.rfind('.');

		path = sTmp.substr(0, rlen);
		sfile = sTmp.substr(len, rlen - rlen);

		int i = 1;

		while (i < 20 && err != 0) // 修改循环条件
		{
			sprintf_s(sName, "%s_T%d.log", path.c_str(), i);

			// 每次尝试都需要重新初始化 temp_fp
			temp_fp = NULL;
			err = fopen_s(&temp_fp, sName, "a");
			m_fp = temp_fp; // 更新成员变量

			i++;
		}
		if (err != 0) // 所有尝试都失败
		{
			m_fp = NULL; // 确保 m_fp 为 NULL
			return false;
		}
	}
	strcpy_s(m_FileName, sName);
	m_bOpened = true;
	return true;

	/*
	Close();
	char sName[512] = { 0 };
	//strcpy(sName, FileName);
	string sYMD = getYearMouthDay();
	sprintf_s(sName, "%s//%s//%s", logPath, (char*)sYMD.c_str(), FileName);

	// 创建文件夹
	char sPath[512] = { 0 };
	sprintf_s(sPath, "%s//%s//", logPath, (char*)sYMD.c_str());
	if (_access(sPath, 0) == -1) // 没有文件夹创建文件夹
	{
		CreateDirectoryA(sPath, 0);
		sprintf_s(sName, "%s//%s//%s", logPath, (char*)sYMD.c_str(), FileName);
	}

	if (m_FileName == NULL)
		return false;
	m_fp = fopen(sName, "a");
	if (m_fp == NULL)
	{
		string sTmp = sName;
		string path, sfile;

		string::size_type len = sTmp.find('/');
		string::size_type rlen = sTmp.rfind('.');

		path = sTmp.substr(0, rlen);
		sfile = sTmp.substr(len, rlen - rlen);

		int i = 1;

		while (i < 20 && m_fp == NULL)
		{
			//sTmp.Format("%s/%s_T%d.log", path, sfile, i);
			sprintf_s(sName, "%s_T%d.log", path.c_str(), i);
			//strcpy(sName, sTmp);
			m_fp = fopen(sName, "a");
			i++;
		}
		if (m_fp == NULL)
		{
			return false;
		}
	}
	strcpy_s(m_FileName, sName);
	m_bOpened = true;
	return true;
	*/
}

void CFileLogEx::Close(void)
{
	if (m_fp == NULL)
		return;

	if (m_bOpened)
	{
		fclose(m_fp);
		m_fp = NULL;
	}

	m_bOpened = false;
}

void CFileLogEx::LogSystemError(HMODULE hSource, DWORD id)
{
	LPTSTR lpMsgBuf;
	if (hSource) {
		FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_IGNORE_INSERTS,
			hSource, id, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
		AddLog(LOG_ERROR, (LPCTSTR)lpMsgBuf);
	}
	else {
		FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL, id, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
		AddLog(LOG_ERROR, (LPCTSTR)lpMsgBuf);
	}
	LocalFree(lpMsgBuf);
}

BOOL CFileLogEx::AddMessage(const char* s1, const char* s2)
{
	CString s = s1;
	s += s2;
	return AddLog(LOG_MESSAGE, s);
}

BOOL CFileLogEx::AddLog(long LogTypeCode, /*const char *Source, */const char* Content)
{
	if (m_fp == NULL)
		return FALSE;

	if (dataChange())
	{
		delPreLogDir();
		Open((char*)m_logPath.c_str(), m_logName.c_str());
	}

	SYSTEMTIME st;
	::GetLocalTime(&st);
	CString s;
	s.Format("%04d-%02d-%02d %02d:%02d:%02d.%03d ", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);

	if (LogTypeCode == LOG_ERROR)
		s += "ERROR   ";
	else if (LogTypeCode == LOG_MESSAGE)
		s += "MESSAGE ";
	else if (LogTypeCode == LOG_WARNING)
		s += "WARNING ";
	else
		s += "INFO    ";
	//CString t = Source;
	//char sp[13] = "            ";
	//if (t.GetLength() >= 12)
	//{
	//	s += t;
	//	s += ' ';
	//}
	//else
	//{
	//	memcpy(sp, t, t.GetLength());
	//	s += sp;
	//}
	//if (LogTypeCode == LOG_MESSAGE)
	//	s += '\n';
	s += Content;
	s += "\n";
	m_cs.Lock();
	int nRet = fputs(s, m_fp);
	if (nRet != EOF) nRet = fflush(m_fp);
	if (nRet == EOF)
	{
		m_cs.Unlock();
		return FALSE;
	}

	if (ftell(m_fp) > (m_nMaxLogSize * 1024))
	{
		fclose(m_fp);
		Backup(m_nFiles);
		m_bOpened = FALSE;
		BOOL b = true;// = Open(m_FileName);
		m_cs.Unlock();
		return b;
	}
	m_cs.Unlock();

	return TRUE;
}

void CFileLogEx::Backup(long nMax)
{
	CString s = m_FileName, s1, temp;
	s += '.';
	temp.Format("%d", nMax); s += temp;
	remove(s);
	for (int i = nMax; i > 1; i--)
	{
		s = m_FileName; s += '.'; temp.Format("%d", i - 1); s += temp;
		s1 = m_FileName; s1 += '.'; temp.Format("%d", i); s1 += temp;
		rename(s, s1);
	}
	rename(m_FileName, s);
}