#include "main.h"
using namespace cv;
int warp0, warp1 = 0;
int expose0, expose1 = 0;
int m0, m1 = 0;
int T_warp_exp()
{
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	if (dynamic_cast<BlocksCompensator*>(compensator.get()))
	{
		BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
		bcompensator->setNrFeeds(expos_comp_nr_feeds);
		bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
		bcompensator->setBlockSize(expos_comp_block_size,expos_comp_block_size);
	}
	print_mtx.lock();
	LOGLN("----曝光器初始化成功----");
	LOGLN("----曝光开始----");
	print_mtx.unlock();
	while (1)
	{		
		Mat image0,mask0;
		Mat image1, mask1;
		vector<UMat> image_warped(2);
		vector<UMat> mask_warped(2);
		vector<Point> corners(2);
		std::tuple<vector<UMat>, vector<UMat>, vector<Point>> lwc_frame;
		if (!FIFO_rwc_frame_0.empty() && !FIFO_rwc_frame_1.empty())
		{
			unique_lock<mutex> lck_rwc0(rwc_frame_mtx0);
			image_warped[0] = std::get<0>(FIFO_rwc_frame_0.front());
			mask_warped[0]  = std::get<1>(FIFO_rwc_frame_0.front());
			corners[0]      = std::get<2>(FIFO_rwc_frame_0.front());
			FIFO_rwc_frame_0.pop();
			lck_rwc0.unlock();
			
			/*imshow("image0", image_warped[0]);
			imshow("mask0", mask_warped[0]);*/

			unique_lock<mutex> lck_rwc1(rwc_frame_mtx1);
			image_warped[1] = std::get<0>(FIFO_rwc_frame_1.front());
			mask_warped[1]  = std::get<1>(FIFO_rwc_frame_1.front());
			corners[1]      = std::get<2>(FIFO_rwc_frame_1.front());
			FIFO_rwc_frame_1.pop();
			lck_rwc1.unlock();
			
			/*imshow("image1", image_warped[1]);
			imshow("mask1", mask_warped[1]);*/								

			compensator->feed(corners, image_warped, mask_warped);

			vector<UMat> image_warped_c(2);
			vector<UMat> mask_warped_c(2);
			vector<Point> corners_c(2);

			for (int i = 0; i < 2; i++)
			{
				compensator->apply(i, corners[i], image_warped[i], mask_warped[i]);
			}

			for (int i = 0; i < 2; i++)
			{
				image_warped_c[i] =  image_warped[i].clone();
				mask_warped_c[i]  =  mask_warped[i].clone();
				corners_c[i]      =  corners[i];
			}

			lwc_frame = {image_warped_c,mask_warped_c,corners_c };

			unique_lock<mutex> lwc_lck(lwc_frame_mtx);			
			FIFO_lwc_frame.push(lwc_frame);
			lwc_lck.unlock();

			/*print_mtx.lock();
			LOGLN("第" << ++expose0 << "帧曝光完毕");
			LOGLN("第" << ++expose1 << "帧曝光完毕");
			print_mtx.unlock();	*/		
		}

		 else if ((exti_flag0 && exti_flag1) && (FIFO_rwc_frame_0.empty() || FIFO_rwc_frame_1.empty())) 
		{
			break;
		}
			
	}
	expose_flag = true;
	print_mtx.lock();
	LOGLN("----曝光结束----");
	print_mtx.unlock();
	return 0;
}