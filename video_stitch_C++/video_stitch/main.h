#pragma once
#ifndef __main_h__
#define __main_h__

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <fstream>
#include <string>
#include <iostream>
#include <thread>

#include "camera_param.h"
#include "T_cam_0.h"
#include "T_cam_1.h"
#include "T_warp_exp.h"
#include "T_est_seam.h"
#include "T_stitch.h"
#include "T_kp_0.h"
#include "T_kp_1.h"
#include "T_hg.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

#define LOG(msg) cout << msg
#define LOGLN(msg) cout << msg <<endl

extern double warped_image_scale;
extern double work_scale;
extern double seam_scale;
extern double work_megapix;
extern double seam_megapix;
extern double seam_work_aspect;
extern float blend_strength;
extern int kp0_flag;
extern int kp1_flag;
extern int cam_0_i;
extern int cam_1_i;
extern int camera_nums;
extern int expos_comp_type;
extern int expos_comp_nr_feeds;
extern int expos_comp_nr_filtering;
extern int expos_comp_block_size;
extern int blender_type;

extern bool seam_flag;
extern bool expose_flag;
extern bool exti_flag0;
extern bool exti_flag1;
extern bool cam_0_ret;
extern bool cam_1_ret;
extern bool is_work_scale_set;
extern bool is_seam_scale_set;

extern condition_variable org_frame_cv;
extern condition_variable find_kp0_cv;
extern condition_variable find_kp1_cv;

extern mutex org_frame_mtx;
extern mutex org_frame_mtx0;
extern mutex org_frame_mtx1;
extern mutex rwc_frame_mtx0;
extern mutex rwc_frame_mtx1;
extern mutex lwc_frame_mtx;
extern mutex exp_frame_mtx;
extern mutex seam_frame_mtx;
extern mutex set_mtx;
extern mutex print_mtx;
extern mutex kp_mtx0;
extern mutex kp_mtx1;

extern Ptr<Blender>   blender;
extern Ptr<Feature2D> finder;
extern Ptr<FeaturesMatcher> matcher;
extern Ptr<SeamFinder> seam_finder;
extern cv::Ptr<cv::detail::RotationWarper> warper0;
extern cv::Ptr<cv::detail::RotationWarper> warper1;

extern queue<Mat> FIFO_org_frame_0;
extern queue<Mat> FIFO_org_frame_1;
extern queue<ImageFeatures> FIFO_kp_0;
extern queue<ImageFeatures> FIFO_kp_1;
extern queue<tuple<UMat, UMat, Point>> FIFO_rwc_frame_0;
extern queue<tuple<UMat, UMat, Point>> FIFO_rwc_frame_1;
extern queue<tuple<vector<UMat>, vector<UMat>, vector<Point>>> FIFO_lwc_frame;
extern queue<tuple<vector<UMat>, vector<UMat>, vector<Point>>> FIFO_seam_frame;

extern vector<MatchesInfo> pairwise_matches;
extern vector<CameraParams> camera;


#endif // !__global_h__

