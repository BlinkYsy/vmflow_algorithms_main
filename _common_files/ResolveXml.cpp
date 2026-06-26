//#include "pch.h"
#include "ResolveXml.h"
#include "Util.h"
#include <sstream>

string ResolveXml::float2string(float Num)
{
	std::ostringstream oss;
	oss << Num;
	string str(oss.str());
	return str;
}

string ResolveXml::cvRect2string(cv::Rect rect)
{
	std::ostringstream oss;
	oss << rect.x << "," << rect.y << "," << rect.width << "," << rect.height;
	string str(oss.str());
	return str;
}

string ResolveXml::cvPts2string(vector<cv::Point>& vePt)
{
	string ptstring = "";
	for (size_t i = 0; i < vePt.size(); i++)
	{
		std::ostringstream ossx;
		ossx << vePt[i].x;
		ptstring += (ossx.str()) + ",";

		std::ostringstream ossy;
		ossy << vePt[i].y;
		ptstring += (ossy.str()) + ";";
	}

	CString str = ptstring.c_str();
	str = str.Left(str.GetLength() - 1);

	return (string)(char*)str.GetBuffer();
}

string ResolveXml::parseXmlInterfaceName(std::string strBody, string param)
{
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return "";

	CString interfaceName = "-1";

	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		xml.FindNode(1);
		interfaceName = xml.GetTagName();
		break;
	}

	return (char*)interfaceName.GetBuffer();
}

string ResolveXml::parseXmlStr(std::string strBody, string param, string item, string childItem)
{
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return "";

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childItem.c_str());
			strSN = xml.GetData();
		}
	}

	return (char*)strSN.GetBuffer();
}

int ResolveXml::parseXmlInt(std::string strBody, string param, string item,string childItem)
{
	//printf("strBody %s\n", strBody.c_str());
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childItem.c_str());
			strSN = xml.GetData();
		}
	}

	return atoi((char*)strSN.GetBuffer());
}

double ResolveXml::parseXmlDou(std::string strBody, string param, string item, string childItem)
{
	//printf("strBody %s\n", strBody.c_str());
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childItem.c_str());
			strSN = xml.GetData();
		}
	}

	return atof((char*)strSN.GetBuffer());
}

int ResolveXml::parseXmlDouble(string strBody, string param, string item, string childItem, vector<double>& veDouble)
{
	veDouble.clear();
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childItem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	for (int i = 0; i < vePtStr.size(); i++)
	{
		float x = atof(vePtStr[i].c_str());
		veDouble.push_back(x);
	}

	return 0;
}

int ResolveXml::parseXmlVePts2f(string strBody, string param, string item, string childitem, std::vector<cv::Point2f>& vePts)
{
	vePts.clear();
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	for (int i = 0; i < vePtStr.size(); i++)
	{
		vector<string> vePtCor;
		util.SplitString(vePtStr[i], vePtCor, ",");
		if (vePtCor.size() >= 2)
		{
			float x = atof(vePtCor[0].c_str());
			float y = atof(vePtCor[1].c_str());
			vePts.push_back(cv::Point2f(x, y));
		}
	}

	return 0;
}

int ResolveXml::parseXmlVePts(string strBody, string param, string item, string childitem,vector<cv::Point>& vePts)
{
	vePts.clear();
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	for (int i = 0; i < vePtStr.size(); i++)
	{
		vector<string> vePtCor;
		util.SplitString(vePtStr[i], vePtCor, ",");
		if (vePtCor.size() >= 2)
		{
			int x = atoi(vePtCor[0].c_str());
			int y = atoi(vePtCor[1].c_str());
			vePts.push_back(cv::Point(x, y));
		}
	}

	return 0;
}

int ResolveXml::parseXmlPt(string strBody, string param, string item, string childitem, cv::Point& pt)
{
	//vePts.clear();
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ",");

	int x = 0;
	int y = 0;
	if (vePtStr.size() == 2)
	{
		x = atoi(vePtStr[0].c_str());
		y = atoi(vePtStr[1].c_str());
	}

	pt.x = x;
	pt.y = y;

	return 0;
}

int ResolveXml::parseXmlPt2d(string strBody, string param, string item, string childitem, cv::Point2d& pt)
{
	//vePts.clear();
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ",");

	float x = atof(vePtStr[0].c_str());
	float y = atof(vePtStr[1].c_str());

	pt.x = x;
	pt.y = y;

	return 0;
}

int ResolveXml::parseXmlRect(string strBody, string param, string item, string childitem, cv::Rect& rect)
{
	rect = cv::Rect(0, 0, 0, 0);
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	for (int i = 0; i < vePtStr.size(); i++)
	{
		vector<string> vePtCor;
		util.SplitString(vePtStr[i], vePtCor, ",");
		if (vePtCor.size() == 4)
		{
			int x = atoi(vePtCor[0].c_str());
			int y = atoi(vePtCor[1].c_str());
			int w = atoi(vePtCor[2].c_str());
			int h = atoi(vePtCor[3].c_str());
			rect = cv::Rect(x,y,w,h);
			//vePts.push_back(cv::Point(x, y));
		}
		else
			return -1;
	}

	return 0;
}

int ResolveXml::parseXmlVecRectROI(string strBody, string param, string item, string childitem, vector<cv::Rect>& vecRectROI)
{
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	for (int i = 0; i < vePtStr.size(); i++)
	{
		vector<string> vePtCor;
		util.SplitString(vePtStr[i], vePtCor, ",");
		if (vePtCor.size() == 4)
		{
			float X = atoi(vePtCor[0].c_str());
			float Y = atoi(vePtCor[1].c_str());
			float width = atoi(vePtCor[2].c_str());
			float height = atoi(vePtCor[3].c_str());
			Rect Rect;
			Rect.x = X;
			Rect.y = Y;
			Rect.width = width;
			Rect.height = height;
			vecRectROI.push_back(Rect);
		}
	}
	return 0;
}

int ResolveXml::parseXmlRectROI(string strBody, string param, string item, string childitem, vector<cv::RotatedRect>& vecRectROI)
{
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	for (int i = 0; i < vePtStr.size(); i++)
	{
		vector<string> vePtCor;
		util.SplitString(vePtStr[i], vePtCor, ",");
		if (vePtCor.size() == 5)
		{
			float X = atoi(vePtCor[0].c_str());
			float Y = atoi(vePtCor[1].c_str());
			float width = atoi(vePtCor[2].c_str());
			float height = atoi(vePtCor[3].c_str());
			float angle = atoi(vePtCor[4].c_str());
			RotatedRect rotatedRect;
			rotatedRect.center.x = X;
			rotatedRect.center.y = Y;
			rotatedRect.size.width = width;
			rotatedRect.size.height = height;
			rotatedRect.angle = angle;
			vecRectROI.push_back(rotatedRect);
		}
	}
	return 0;
}

int ResolveXml::parseXmlCircleROI(string strBody, string param, string item, string childitem, vector<circleROI>& vecCircleROI)
{
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	for (int i = 0; i < vePtStr.size(); i++)
	{
		vector<string> vePtCor;
		util.SplitString(vePtStr[i], vePtCor, ",");
		if (vePtCor.size() == 6)
		{
			float centerX = atoi(vePtCor[0].c_str());
			float centerY = atoi(vePtCor[1].c_str());
			float outerRadius = atoi(vePtCor[2].c_str());
			float innerRadius = atoi(vePtCor[3].c_str());
			float startAngle = atoi(vePtCor[4].c_str());
			float angleRange = atoi(vePtCor[5].c_str());

			circleROI circleroi;
			circleroi.centerX = centerX;
			circleroi.centerY = centerY;
			circleroi.outerRadius = outerRadius;
			circleroi.innerRadius = innerRadius;
			circleroi.startAngle = startAngle;
			circleroi.angleRange = angleRange;
			vecCircleROI.push_back(circleroi);
		}
	}
	return 0;
}

int ResolveXml::parseXmlPolygonROI(string strBody, string param, string item, string childitem, vector<vector<cv::Point>>& vecPolygonROI)
{
	vector<cv::Point> vecpolygonroi;
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	for (int i = 0; i < vePtStr.size(); i++)
	{
		vector<string> vePtCor;
		util.SplitString(vePtStr[i], vePtCor, ",");
		if (vePtCor.size() % 2 == 0 && vePtCor.size() >= 6)
		{
			for (size_t j = 0; j < vePtCor.size(); j = j + 2)
				vecpolygonroi.push_back(Point(atoi(vePtCor[j].c_str()), atoi(vePtCor[j + 1].c_str())));
			vecPolygonROI.push_back(vecpolygonroi);
			vecpolygonroi.clear();
		}
	}
	return 0;
}

int ResolveXml::parseXmlMaskCropAreas(string strBody, string param, string item, string childitem, cv::Rect& rect,int& selectShape, int& selectColor)
{
	rect = cv::Rect(0, 0, 0, 0);
	bool bRet = true;
	CMarkup xml;
	BOOL bFind = true;
	//bFind = xml.Load((MCD_CSTR)strBody.c_str());
	bFind = xml.SetDoc(strBody.c_str());

	if (bFind == false)
		return -1;

	CString strSN = "-1";
	while (xml.FindElem((MCD_CSTR)param.c_str()))
	{
		xml.IntoElem();
		while (xml.FindElem((MCD_CSTR)item.c_str()))
		{
			xml.IntoElem();
			xml.FindElem((MCD_CSTR)childitem.c_str());
			strSN = xml.GetData();
		}
	}

	strSN += ";";
	// 解析点集
	Util util;
	vector<string> vePtStr;
	string splitStr = (string)(char*)strSN.GetBuffer();
	util.SplitString(splitStr, vePtStr, ";");
	if (vePtStr.size() != 1)
		return -1;
	
	vector<string> vePtCor;
	util.SplitString(vePtStr[0], vePtCor, ",");
	if (vePtCor.size() == 6)
	{
		selectShape = atoi(vePtCor[0].c_str());
		selectColor = atoi(vePtCor[1].c_str());
		int x = atoi(vePtCor[2].c_str());
		int y = atoi(vePtCor[3].c_str());
		int w = atoi(vePtCor[4].c_str());
		int h = atoi(vePtCor[5].c_str());
		
		rect = cv::Rect(x, y, w, h);
		//vePts.push_back(cv::Point(x, y));
	}
	else
		return -2;
	
	return 0;
}

int ResolveXml::saveLocalCircleXml(string& xmlOut, string funName, cv::Point2f center, double radius, std::vector<cv::Point> contourPts)
{
	

	return 0;
}

int ResolveXml::saveLocalMatchXml(string& xmlOut, string funName,vector<string> matchResult,cv::Point pt)
{
	CMarkup xml;
	xml.SetDoc("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n");
	xml.AddElem((MCD_CSTR)"Param");						//在当前主位置元素或最后兄弟位置之后增加一个元素
	xml.IntoElem();										//进入当前主位置的下一级，当前的位置变为父位置。

	xml.AddElem((MCD_CSTR)funName.c_str());						//在当前主位置元素或最后兄弟位置之后增加一个元素
	xml.IntoElem();										//进入当前主位置的下一级，当前的位置变为父位置。

	string resultStr = "";
	for (int i = 0; i < matchResult.size(); i++)
	{
		resultStr += matchResult[i];
	}

	CString str = resultStr.c_str();
	str = str.Left(str.GetLength() - 1);

	xml.AddElem((MCD_CSTR)"matchResult", (MCD_CSTR)str.GetBuffer());

	str.Format("%d,%d", pt.x, pt.y);
	xml.AddElem((MCD_CSTR)"matchCenterPoint", (MCD_CSTR)str.GetBuffer());

	xml.OutOfElem();										//使当前父位置变成当前位置。

	CString Out = xml.GetDoc();
	xmlOut = (char*)Out.GetBuffer();

	return 0;
}
