//#include "pch.h"
#include "renderParam.h"
#include "ResolveXml.h"
#include <sstream>

string renderParam::conRenderRect()
{
	string ptstring = "";
	for (size_t i = 0; i < m_renderRectBox.size(); i++)
	{
		std::ostringstream oss;
		oss << m_renderRectBox[i].center.x << "," << m_renderRectBox[i].center.y << ","\
			<< m_renderRectBox[i].size.width << "," << m_renderRectBox[i].size.height << ","\
			<< m_renderRectBox[i].angle << ";";
		ptstring += (oss.str());
	}
	CString str = ptstring.c_str();
	str = str.Left(str.GetLength() - 1);

	return (string)(char*)str.GetBuffer();
}

string renderParam::conRenderCircle()
{
	string ptstring = "";
	for (size_t i = 0; i < m_renderCircle.size(); i++)
	{
		std::ostringstream oss;
		oss << m_renderCircle[i].centerX << "," << m_renderCircle[i].centerY << ","\
			<< m_renderCircle[i].outerRadius << ";";
		ptstring += (oss.str());
	}
	CString str = ptstring.c_str();
	str = str.Left(str.GetLength() - 1);

	return (string)(char*)str.GetBuffer();
}

string renderParam::conRenderPolygon()
{
	string ptstring = "";
	for (size_t i = 0; i < m_renderPolygon.size(); i++)
	{
		for (size_t j = 0; j < m_renderPolygon[i].size(); j++)
		{
			std::ostringstream oss;
			oss << m_renderPolygon[i][j].x << "," << m_renderPolygon[i][j].y << ",";
			ptstring += (oss.str());
		}
		ptstring = ptstring.erase(ptstring.size() - 1, 1);
		ptstring += ";";
	}
	CString str = ptstring.c_str();
	str = str.Left(str.GetLength() - 1);

	return (string)(char*)str.GetBuffer();
}

string renderParam::conRenderDot()
{
	string ptstring = "";
	for (size_t i = 0; i < m_renderDot.size(); i++)
	{
		std::ostringstream oss;
		oss << m_renderDot[i].x << "," << m_renderDot[i].y << ";";
		ptstring += (oss.str());
	}
	CString str = ptstring.c_str();
	str = str.Left(str.GetLength() - 1);

	return (string)(char*)str.GetBuffer();
}

string renderParam::conRenderLine()
{
	string ptstring = "";
	for (size_t i = 0; i < m_renderLine.size(); i++)
	{
		std::ostringstream oss;
		oss << m_renderLine[i].linePt1.x << "," << m_renderLine[i].linePt1.y << ","\
			<< m_renderLine[i].linePt2.x << "," << m_renderLine[i].linePt2.y << ";";
		ptstring += (oss.str());
	}
	CString str = ptstring.c_str();
	str = str.Left(str.GetLength() - 1);

	return (string)(char*)str.GetBuffer();
}

string renderParam::conRenderDelDot()
{
	string ptstring = "";
	for (size_t i = 0; i < m_renderDelDot.size(); i++)
	{
		std::ostringstream oss;
		oss << m_renderDelDot[i].x << "," << m_renderDelDot[i].y << ";";
		ptstring += (oss.str());
	}
	CString str = ptstring.c_str();
	str = str.Left(str.GetLength() - 1);

	return (string)(char*)str.GetBuffer();
}

void renderParam::conOutImg(cv::Mat inImg, void*& outImg)
{
	int size = inImg.total() * inImg.elemSize();
	unsigned char* dstMat = new unsigned char[size];
	std::memcpy(dstMat, inImg.data, size * sizeof(unsigned char));
	outImg = new cv::Mat(inImg.rows, inImg.cols, inImg.type(), dstMat);
}

