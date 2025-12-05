#include "main.h"
using namespace std;
using namespace cv;
int cam_1_i = 0;
bool cam_1_ret = false;

int T_cam_1()
{	
	bool is_size_set = false;
	VideoCapture video("D:\\DeskTop\\file\\ptest\\video1.mp4");
	if (!video.isOpened())
	{
		LOGLN("----相机1加载失败----");
		return -1;
	}
	Mat frame;
	UMat frame_warped, mask, mask_warped;
	Point corner;
	std::tuple<UMat, UMat, Point> image_info;
	LOGLN("----相机1开始播放----");
	while (1)
	{		
		cam_1_ret = video.read(frame);
		if (!cam_1_ret)
		{
			exti_flag1 = true;
			print_mtx.lock();
			LOGLN("----相机1播放结束----");
			print_mtx.unlock();
			break;
		}		
		Mat K;		
		camera[1].K().convertTo(K, CV_32F);

		mask.create(frame.size(), CV_8U);
		mask.setTo(Scalar::all(255));
		
		corner = warper1->warp(frame, K, camera[1].R, INTER_LINEAR, BORDER_REFLECT, frame_warped);
		warper1->warp(mask, K, camera[1].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
		
		image_info = { frame_warped.clone() ,mask_warped.clone() ,corner };			

		/*unique_lock<mutex> lck_org(org_frame_mtx1);
		FIFO_org_frame_1.push(frame.clone());
		lck_org.unlock();*/

		unique_lock<mutex> lck_rwc(rwc_frame_mtx1);
		FIFO_rwc_frame_1.push(image_info);
		lck_rwc.unlock();

		
		print_mtx.lock();
		LOGLN("相机1第"<<++cam_1_i<<"帧读取完毕");
		LOGLN("right corner:" << corner);
		print_mtx.unlock();			
	}
	video.release();
	return 0;
}