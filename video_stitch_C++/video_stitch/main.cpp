#include "main.h"
using namespace cv;
using namespace std;

bool have_cam_param = false;

bool exti_flag0  = false;
bool exti_flag1  = false;
bool expose_flag = false;
bool seam_flag   = false;
bool is_work_scale_set = false;
bool is_seam_scale_set = false;
bool open_thread = true;
int kp0_flag = 0;
int kp1_flag = 0;
int camera_nums = 2;
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
int blender_type = Blender::MULTI_BAND;

float blend_strength = 20;
double warped_image_scale;
double work_scale;
double seam_scale;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double seam_work_aspect;

mutex org_frame_mtx0;
mutex org_frame_mtx1;
mutex rwc_frame_mtx0;
mutex rwc_frame_mtx1;
mutex print_mtx;
mutex set_mtx;
mutex kp_mtx0;
mutex kp_mtx1;
mutex lwc_frame_mtx;
mutex seam_frame_mtx;
mutex exp_frame_mtx;

condition_variable org_frame_cv;
condition_variable find_kp0_cv;
condition_variable find_kp1_cv;

queue<Mat> FIFO_org_frame_0;
queue<Mat> FIFO_org_frame_1;
queue<ImageFeatures> FIFO_kp_0;
queue<ImageFeatures> FIFO_kp_1;
queue<tuple<UMat, UMat, Point>> FIFO_rwc_frame_0;
queue<tuple<UMat, UMat, Point>> FIFO_rwc_frame_1;
queue<tuple<vector<UMat>, vector<UMat>, vector<Point>>> FIFO_lwc_frame;
queue<tuple<vector<UMat>, vector<UMat>, vector<Point>>> FIFO_seam_frame;

Ptr<Feature2D> finder = xfeatures2d::SURF::create();
Ptr<FeaturesMatcher> matcher = makePtr<BestOf2NearestMatcher>(true, 0.65f);
cv::Ptr<cv::detail::RotationWarper> warper0;
cv::Ptr<cv::detail::RotationWarper> warper1;
Ptr<SeamFinder> seam_finder;
Ptr<Blender>    blender;
vector<MatchesInfo> pairwise_matches;
vector<CameraParams> camera;


//string filename = "D:\\DeskTop\\file\\stereo_calib1.yaml";

int main()
{

	auto start = std::chrono::high_resolution_clock::now();

	camera_param_init();	
	warper_init();
	thread test_thread0(T_cam_0);
	thread test_thread1(T_cam_1);
	thread test_thread2(T_warp_exp);
	thread test_thread3(T_est_seam);
	thread test_thread4(T_stitch);
#if have_cam_param
	thread test_thread2(T_kp_0);
	thread test_thread3(T_kp_1);
	thread test_thread4(T_hg);
	//cout << cam_0_i << "and" << cam_1_i << endl;
	test_thread2.join();
	test_thread3.join();
	test_thread4.join();
#endif
	test_thread0.join();
	test_thread1.join();
	test_thread2.join();
	test_thread3.join();
	test_thread4.join();	
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	LOGLN("stitch time:" << duration.count() << "sec");
	/*camera_param_init();
	warper_init();
	Point2f pt(0,0);
	Point2f corner;
	Mat K = camera[1].K();
	K.convertTo(K,CV_32F);
	//corner = warper0->warpPointBackward(pt,K,camera[1].R);
	corner = warper0->warpPoint(pt, K, camera[1].R);
	cout << corner << endl;*/
	return 0;
}