#include "main.h"
using namespace cv;
using namespace std;
int stitch_num = 0;
bool use_cuda = false;
int T_stitch()
{
	vector<UMat>  image_warped(2), mask_warped(2);
	vector<Point> corner(2);
	vector<Size>  size(2);
	blender = Blender::createDefault(blender_type, use_cuda);
	print_mtx.lock();
	LOGLN("----融合器创建成功----");
	print_mtx.unlock();
	while (1)
	{
		if (!FIFO_seam_frame.empty())
		{
			bool blend_creat = false;

			unique_lock<mutex> seam_lck(seam_frame_mtx);
			image_warped = get<0>(FIFO_seam_frame.front());
			mask_warped = get<1>(FIFO_seam_frame.front());
			corner = get<2>(FIFO_seam_frame.front());
			FIFO_seam_frame.pop();
			seam_lck.unlock();

			for (int i = 0; i < 2; i++)
			{
				size[i] = image_warped[i].size();
			}

			for (int i = 0; i < 2; i++)
			{
				Mat seam_mask, blend_mask;
				blend_mask = mask_warped[i].getMat(ACCESS_RW).clone();
				dilate(mask_warped[i], seam_mask, Mat());
				blend_mask = blend_mask & seam_mask;

				Mat blend_image;
				blend_image = image_warped[i].getMat(ACCESS_RW).clone();
				blend_image.convertTo(blend_image, CV_16S);

				if (!blend_creat)
				{
					Size dst_size = resultRoi(corner, size).size();
					float blend_width = sqrt(static_cast<float>(dst_size.area()) * blend_strength / 100.f);
					if (dynamic_cast<MultiBandBlender*> (blender.get()))
					{
						
						MultiBandBlender* mb = dynamic_cast<MultiBandBlender*> (blender.get());
						mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
					}
					blender->prepare(corner, size);
					blend_creat = true;
				}

				blender->feed(blend_image, blend_mask, corner[i]);
			}
			/*print_mtx.lock();
			LOGLN("第"<<++stitch_num<<"帧融合完毕");
			print_mtx.unlock();*/
			Mat result_image, result_mask;

			if (use_cuda)
			{
				cuda::GpuMat  result_gpu, mask_gpu;		
				auto start = std::chrono::high_resolution_clock::now();
				blender->blend(result_gpu, mask_gpu);
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> duration = end - start;
				LOGLN("blend time:" << duration.count() << "sec");
				result_gpu.download(result_image);
				mask_gpu.download(result_mask);
			}
			else
			{
				auto start = std::chrono::high_resolution_clock::now();
				blender->blend(result_image, result_mask);
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> duration = end - start;
				LOGLN("blend time:" << duration.count() << "sec");
			}									
			result_image.convertTo(result_image, CV_8U);
			imshow("result_image.jpg", result_image);
			waitKey(1);
		}
		else if (seam_flag == true && FIFO_seam_frame.empty())
		{
			break;
		}
	}
	print_mtx.lock();
	LOGLN("----融合结束----");
	print_mtx.unlock();
	return 0;

}