#include "main.h"
using namespace std;
using namespace cv;
int seam_num = 0;
int ii = 4;
int T_est_seam()
{
	tuple<vector<UMat>, vector<UMat>, vector<Point>> lwc_frame;
	vector<UMat> image_warped(2), mask_warped(2);
	vector<Point> corner(2);
	seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);

	print_mtx.lock();
	LOGLN("----拼接缝查找器初始化成功----");	
	print_mtx.unlock();

	while(1)
	{
		if (!FIFO_lwc_frame.empty()) 
		{
			unique_lock<mutex> lwc_lck(lwc_frame_mtx);
			image_warped = get<0>(FIFO_lwc_frame.front());
			mask_warped  = get<1>(FIFO_lwc_frame.front());
			corner = get<2>(FIFO_lwc_frame.front());
			FIFO_lwc_frame.pop();
			lwc_lck.unlock();

			for (int i = 0; i < 2; i++)
			{
				image_warped[i].convertTo(image_warped[i], CV_32F);
			}
			ii++;
			if (ii >= 5)
			{
				seam_finder->find(image_warped, corner, mask_warped);
				ii = 0;
			}											
			vector<UMat> image_warped_c(2);
			vector<UMat> mask_warped_c(2);
			vector<Point> corner_c(2);
			for (int i = 0; i < 2; i++)
			{
				image_warped_c[i] = image_warped[i].clone();
				mask_warped_c[i]  = mask_warped[i].clone();
				corner_c[i] = corner[i];
			}

			lwc_frame = { image_warped_c ,mask_warped_c ,corner_c };
			unique_lock<mutex> seam_lck(seam_frame_mtx);
			FIFO_seam_frame.push(lwc_frame);
			seam_lck.unlock();

			/*print_mtx.lock();
			LOGLN("第"<<++seam_num<<"帧拼接缝查找完毕");
			print_mtx.unlock();*/
			//imshow("mask_warped[0]", mask_warped[0]);
			//imshow("mask_warped[1]", mask_warped[1]);
			//waitKey(1);
		}
		else if(expose_flag==true && FIFO_lwc_frame.empty())
		{
			break;
		}
		
	}
	seam_flag = true;
	print_mtx.lock();
	LOGLN("----拼接缝查找结束----");
	print_mtx.unlock();
	return 0;
}