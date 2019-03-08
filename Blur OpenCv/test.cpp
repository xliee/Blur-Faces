#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int sumRGB[3];
int avgRGB[3];
int blurxd(17);

CascadeClassifier cascade;
const std::string LOAD_PATH = "C:\\Users\\frangg16\\Pictures\\test.jpeg";
array<String, 2> CASCADE_FILE { "W:\\Documents\\Web\\c++\\face_detect_n_track-master\\haarcascade_frontalface_default.xml", "W:\\Documents\\Web\\c++\\face_detect_n_track-master\\haarcascade_frontalface_default.xml" };
const cv::String    CASCADE_FILE2("W:\\Documents\\Web\\c++\\face_detect_n_track-master\\haarcascade_frontalface_default.xml");
int xPointOne(-1), xPointTwo(-1), yPointOne(-1), yPointTwo(-1);
Mat frame;


void findAverage(const int x, const int y)
{
	for (int i = 0; i < blurxd; i++)
	{
		for (int j = 0; j < blurxd; j++)
		{
			if ((y + i) < frame.rows && (x + j) < frame.cols)
			{
				sumRGB[0] += frame.ptr<uchar>(y + i)[3 * (x + j)];
				sumRGB[1] += frame.ptr<uchar>(y + i)[3 * (x + j) + 2];
				sumRGB[2] += frame.ptr<uchar>(y + i)[3 * (x + j) + 4];
			}
		}
	}

	for (int k = 0; k < 3; k++)
		avgRGB[k] = sumRGB[k] / (blurxd * blurxd);
} // end findAverage

void blurRegion(const int x, const int y)
{
	for (int i = 0; i < blurxd; i++)
	{
		for (int j = 0; j < blurxd; j++)
		{
			if ((y + i) < frame.rows && (x + j) < frame.cols)
			{
				frame.ptr<uchar>(y + i)[3 * (x + j)] = avgRGB[0];
				frame.ptr<uchar>(y + i)[3 * (x + j) + 2] = avgRGB[1];
				frame.ptr<uchar>(y + i)[3 * (x + j) + 4] = avgRGB[2];
			}
		}
	}
} // end blurRegion
void blur()
{
	for (int i = yPointOne; i < yPointTwo; i += blurxd)
	{
		for (int j = xPointOne; j < xPointTwo; j += blurxd)
		{
			findAverage(j, i);
			blurRegion(j, i);

			for (int k = 0; k < 3; k++)
				avgRGB[k] = sumRGB[k] = 0;
		}
	}
} // end blur
void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale) {

	vector<Rect> faces;
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	for (int i = 0; i < CASCADE_FILE.size(); i++) {
		cascade.load(CASCADE_FILE[i]);
	

		
		try
		{
			cvtColor(img, gray, COLOR_BGR2GRAY);
			cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(15, 15));
			for (size_t i = 0; i < faces.size(); i++)
			{
				Rect r = faces[i];
				Scalar color = Scalar(255, 0, 0);
				xPointOne = cvRound(r.x*scale);
				xPointTwo = cvRound((r.x + r.width - 1)*scale);
				yPointOne = cvRound(r.y*scale);
				yPointTwo = cvRound((r.y + r.height - 1)*scale);
				blur();
				Mat im2 = img;
				//rectangle(im2, Point(cvRound(r.x*scale), cvRound(r.y*scale)), Point(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)), color, 0, 0, 0);
				imshow("Face Detection", im2);
				int key2 = waitKey();
			}
		}
		catch (cv::Exception& e)
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}
	}
	imshow("Face Detection", img);
	int key2 = waitKey();

}

int main()
{
	// Load the cascade classifier
	//cascade.load(CASCADE_FILE);
	double scale = 1;
	frame = imread(LOAD_PATH, IMREAD_COLOR);
	detectAndDraw(frame, cascade, scale);
	return 0;
}




