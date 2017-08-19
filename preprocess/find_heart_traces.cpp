#include <stdio.h>  
#include <sys/types.h>  
#include <io.h> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>  
#include<string>
#include<direct.h>
#include<algorithm>

using namespace cv;
using namespace std;


bool cmp(const vector<Point> c1, const vector<Point> c2)
{
	return contourArea(c1) > contourArea(c2);
}


Mat reverse(Mat src)

{

	Mat dst = src<150;

	return dst;
}


void fillHole(const Mat srcBw, Mat &dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

	cv::floodFill(Temp, Point(0, 0), Scalar(255));

	Mat cutImg;//²Ã¼ôÑÓÕ¹µÄÍ¼Ïñ  
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

	dstBw = srcBw | (~cutImg);
}

int main()
{
	int countt = 0;
	struct _finddata_t fileinfo;
	intptr_t hFile = 0;
	string input = "C:\\Users\\Administrator\\Desktop\\az_normal_512x624\\fail\\";
	string current_path = input + "*.jpg";
	if ((hFile = _findfirst(current_path.c_str(), &fileinfo)) == -1)
		exit(1);
	else {
		do {
			/*Process File*/



			Mat src, src_gray;
			Mat grad;

			string output = "C:\\Users\\Administrator\\Desktop\\az_normal_512x624\\outf";

			ifstream ffile;
			ffile.open(output);
			if (!ffile.is_open())
			{
				_mkdir(output.c_str());
			}

			int scale = 1;
			int delta = 0;
			int ddepth = CV_16S;
			int kernel_size = 3;

			/// Load an image
			src = imread(input + fileinfo.name);
			cout << "[Read] " + input + fileinfo.name << endl;
			if (!src.data)
			{
				return -2;
			}

			GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
			cvtColor(src, src_gray, CV_RGB2GRAY);


			/////////////////////////// Sobel ////////////////////////////////////
			/// Generate grad_x and grad_y
			Mat grad_x, grad_y;
			Mat abs_grad_x, abs_grad_y;
			/// Gradient X
			//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
			//Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
			Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
			convertScaleAbs(grad_x, abs_grad_x);
			/// Gradient Y  
			//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
			Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
			convertScaleAbs(grad_y, abs_grad_y);
			/// Total Gradient (approximate)
			addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

			//global 2
			int th = 22;                                                           
			Mat global;
			threshold(grad, global, th, 255, CV_THRESH_BINARY_INV);


			
			Mat element = getStructuringElement(MORPH_RECT, Size(3, 17));
			Mat element2 = getStructuringElement(MORPH_RECT, Size(8, 10));
			/*Mat element3 = getStructuringElement(MORPH_RECT, Size(10, 10));
			Mat element4 = getStructuringElement(MORPH_RECT, Size(5, 5));*/
			Mat out, out2;


			//dilate & erode 
			global = reverse(global);

			dilate(global, out2, element);
			erode(out2, out, element2);


			

			//floodFill
			out = reverse(out);
			Mat dst;
			fillHole(out, dst);




			//contour
			vector<vector<cv::Point>> contours;
			// findcontour 
			cv::findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			//use sort to seek the second largest area  
			double maxArea = 0;
			for (vector<vector<Point>>::iterator i = contours.begin(); i != contours.end();)
			{
				int flag = 0;																
				double area = cv::contourArea(*i);
				if (area < 10000 || area>100000)
					flag = 0;
				else
				{
					for (int j = 0; j < (*i).size(); j++)
					{
						if ((*i)[j].x < 350 && (*i)[j].x >180 && (*i)[j].y > 300 && (*i)[j].y<400)
						{
							flag = 1;
							break;
						}
					}
				}
				if (!flag)
					i = contours.erase(i);
				else
					++i;
			}
			vector<cv::Point> maxContour;
			sort(contours.begin(), contours.end(), cmp);

			int draw = 1;
			if (contours.size() > 1)
				maxContour = contours[1];
			else if (contours.size() > 0)
				maxContour = contours[0];
			else draw = 0;
			vector<vector<Point>> ct;
			ct.push_back(maxContour);
			// convert to rectangles
			//cv::Rect maxRect = cv::boundingRect(maxContour);

			// show area 
			Mat result2(dst.rows, dst.cols, CV_8UC3, Scalar(255, 255, 255));


			Mat result1;
			dst.copyTo(result1);

			for (size_t i = 0; i < contours.size(); i++)
			{
				cv::Rect r = cv::boundingRect(contours[i]);
				cv::rectangle(result1, r, cv::Scalar(80), 10);
			}
			/*cv::imshow("all regions", result1);
			cv::waitKey();*/

			imwrite(output + "\\1\\" + fileinfo.name, result1);

			/*cv::imshow("largest region", result2);
			cv::waitKey();*/


			//curve approximation;
			vector<Point> curve;
			if (draw)
				approxPolyDP(ct[0], curve, 10, 1);
			ct.clear();
			ct.push_back(curve);



			if (draw)
			{

				/*string filename = fileinfo.name;
				string fname = filename.substr(0, filename.size() - 3);
				fname += "jpg";*/

				cv::drawContours(result2, ct, -1, cv::Scalar(80), 3);
				imwrite(output + "\\" + fileinfo.name, result2);
				cout << "[Save] " << output + "\\" + fileinfo.name << ".  count == " << ++countt << endl;
				ofstream tout(output + "\\" + fileinfo.name + ".txt");
				for (vector<Point>::iterator it = curve.begin(); it != curve.end(); it++)
					tout << it->x << "," << it->y << endl;
				tout.close();
			}
			else
				imwrite(output + "\\" + fileinfo.name, src);




		} while (_findnext(hFile, &fileinfo) != -1);
	}
	_findclose(hFile);

	system("pause");
	waitKey(0);
	return 0;
}
