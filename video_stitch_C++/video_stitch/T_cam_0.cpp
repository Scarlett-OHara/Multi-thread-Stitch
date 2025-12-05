#include "main.h"
using namespace cv;
using namespace std;
int cam_0_i = 0;
bool cam_0_ret = false;

int T_cam_0()
{
	bool is_size_set = false;
	VideoCapture video("D:\\DeskTop\\file\\ptest\\video0.mp4");
	if (!video.isOpened())
	{
		LOGLN("----相机0加载失败----");
		return -1;
	}
	Mat frame;
	UMat frame_warped, mask, mask_warped;
	Point corner;
	std::tuple<UMat, UMat, Point> image_info;
	LOGLN("----相机0开始播放----");
	while (1)
	{
		cam_0_ret = video.read(frame);
		if (!cam_0_ret)
		{
			exti_flag0 = true;

			print_mtx.lock();
			LOGLN("----相机0播放结束----");
			print_mtx.unlock();

			break;
		}
		Mat K;		
		mask.create(frame.size(), CV_8U);
		mask.setTo(Scalar::all(255));

		camera[0].K().convertTo(K, CV_32F);
		cv::Rect roi(0, 0, 503, 417);
		corner = warper0->warp(frame, K, camera[0].R, INTER_LINEAR, BORDER_REFLECT, frame_warped);
		warper0->warp(mask, K, camera[0].R, INTER_NEAREST, BORDER_CONSTANT,  mask_warped);
		
		image_info = { frame_warped.clone() ,mask_warped.clone() ,corner };

		unique_lock<mutex> lck_rwc(rwc_frame_mtx0);
		FIFO_rwc_frame_0.push(image_info);
		lck_rwc.unlock();
		
		print_mtx.lock();
		LOGLN("相机0第" << ++cam_0_i << "帧读取完毕");
		LOGLN("left corner:" << corner);
		print_mtx.unlock();			
	}
	video.release();
	return 0;
}