#include "main.h"
int camera_param_init()
{
	WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
	camera.resize(2);
	string filename = "D:\\DeskTop\\file\\stereo_calib1.yaml";
	FileStorage fs(filename,FileStorage::READ);
	if (!fs.isOpened())
	{
		LOGLN("----配置文件打开失败----");
		return -1;
	}
	else
	{
		LOGLN("----配置文件打开成功----");
	}
	Mat R, T;
	Mat camera_matrix_left, camera_matrix_right;
	R = Mat::eye(3, 3, CV_32F);
	T = Mat::zeros(3, 1, CV_32F);
	camera[0].R = R.clone();
	camera[0].t = T.clone();
	fs["camera_matrix_left"] >> camera_matrix_left;
	camera[0].focal = camera_matrix_left.at<double>(0,0);
	camera[0].ppx = camera_matrix_left.at<double>(0, 2);
	camera[0].ppy = camera_matrix_left.at<double>(1, 2);
	camera[0].aspect = camera_matrix_left.at<double>(1, 1) /camera[0].focal;
	LOGLN("----左相机读取完毕----");
	//Mat camera_matrix_right;	
	fs["R"] >> R;
	fs["T"] >> T;
	R.convertTo(R, CV_32F);
	T.convertTo(T, CV_32F);
	camera[1].R = R.clone();
	camera[1].t = T.clone();
	fs["camera_matrix_right"] >> camera_matrix_right;
	camera[1].focal = camera_matrix_right.at<double>(0, 0);
	camera[1].ppx = camera_matrix_right.at<double>(0, 2);
	camera[1].ppy = camera_matrix_right.at<double>(1, 2);
	camera[1].aspect = camera_matrix_right.at<double>(1, 1) / camera[1].focal;
	LOGLN("----右相机读取完毕----");
	vector<Mat> rmats;
	for (int i = 0; i < camera.size();i++)
	{
		rmats.push_back(camera[i].R.clone());
	}

	waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);

	for (int i = 0; i < camera.size(); i++)
	{
		camera[i].R = rmats[i].clone();
	}
	LOGLN("----相机波矫正完毕----");
	for (int i = 0; i < camera.size(); i++) 
	{
		Mat K;
		camera[i].K().convertTo(K, CV_32F);
		camera[i].K() = K.clone();
	}

	warped_image_scale = 0.5 * camera[0].focal + 0.5 * camera[1].focal;	
	//warped_image_scale = 297.083740234375;
	LOGLN("----相机参数初始化完毕----");
	LOGLN("----camera#0:----");
	LOGLN("R");
	LOGLN(camera[0].R);
	LOGLN("K");
	LOGLN(camera[0].K());
	LOGLN("T");
	LOGLN(camera[0].t);
	LOGLN("----camera#1:----");
	LOGLN("R");
	LOGLN(camera[1].R);
	LOGLN("K");
	LOGLN(camera[1].K());
	LOGLN("T");
	LOGLN(camera[1].t);
	return 0;
}
void warper_init()
{
	Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>();
	warper0 = warper_creator->create(static_cast<float> (warped_image_scale));//直接用原始大小图像
	LOGLN("投影器0初始化完毕,缩放系数:"<< warped_image_scale);
	warper1 = warper_creator->create(static_cast<float> (warped_image_scale));
	LOGLN("投影器1初始化完毕,缩放系数:"<< warped_image_scale);
}

