#include "main.h"
using namespace cv;
using namespace cv::detail;
using namespace std;

//double work_scale = sqrt(0.6*1e6/)
int T_kp_0()
{
	this_thread::sleep_for(chrono::milliseconds(100));
	ImageFeatures features;
	//namedWindow("img0", WINDOW_AUTOSIZE);
	while (1)
	{
		unique_lock<mutex> lck_frame(org_frame_mtx0);
		Mat img = FIFO_org_frame_0.front();
		FIFO_org_frame_0.pop();				
		lck_frame.unlock();

		double work_scale = sqrt(0.6 * 1e6 / img.size().area());
		resize(img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
		computeImageFeatures(finder, img, features);

		unique_lock<mutex> lck_kp(kp_mtx0);
		FIFO_kp_0.push(features);
		kp0_flag = 2;
		lck_kp.unlock();
		if (FIFO_org_frame_0.empty()) 
		{
			kp0_flag = 1;
			//find_kp0_cv.notify_all();
			print_mtx.lock();
			cout << "find kp0 end" << endl;
			print_mtx.unlock();
			break;
		}
		/*drawKeypoints(img, features.keypoints, img, Scalar(0, 255, 0));
		imshow("img0", img);
		waitKey(5);*/
	}
	
	return 0;
}