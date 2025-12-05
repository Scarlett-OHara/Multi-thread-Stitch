#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/core/cuda.hpp>
//#include <opencv2/core/utils/parallel.hpp>
#include <iostream>
using namespace std;
using namespace cv;
using namespace xfeatures2d;
int main()
{
	Mat img1, img2;
	img1 = imread("D:\\DeskTop\\file\\ptest\\aloeL.jpg");
	img2 = imread("D:\\DeskTop\\file\\ptest\\aloeR.jpg");
    //创建特征点检测
	Ptr<SIFT> sift = SIFT::create();//描述子对象
	vector<KeyPoint> kp1, kp2;//兴趣点
	Mat describes1, describes2;//描述子
	//寻找兴趣点寻找描述子
	sift->detectAndCompute(img1, Mat(), kp1, describes1);
	sift->detectAndCompute(img2, Mat(), kp2, describes2);
	FlannBasedMatcher matcher;//创建匹配Flann匹配器
	vector<vector<DMatch>> matches;//匹配结果存放
	matcher.knnMatch(describes1, describes2, matches, 2);//存放匹配结果，最匹配，次匹配
	vector<DMatch> good;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < (0.65 * matches[i][1].distance))
		{
			good.push_back(matches[i][0]);
		}
	}
	Mat pic;
	drawMatches(img1, kp1, img2, kp2, good, pic);

	if (good.size() > 10)
	{
		vector<Point2f> src_pts, dst_pts;
		for (const auto match : good)
		{
			src_pts.push_back(kp1[match.queryIdx].pt);
			dst_pts.push_back(kp2[match.trainIdx].pt);
		}

		Mat M, mask;
		M = findHomography(src_pts, dst_pts, RANSAC, 2.0, mask);

		Mat warpimg;
		warpPerspective(img2, warpimg, M.inv(), Size(img1.cols + img2.cols, img2.rows));
		//namedWindow("warpimg", WINDOW_FREERATIO);
		//imshow("warpimg", warpimg);

		
		Mat direct = warpimg.clone();//复制img2变换后的图
		img1.copyTo(direct(Rect(0,0,img1.cols,img1.rows)));//将img1，2放在同一张图上并指定img1放置位置
		//namedWindow("direct", WINDOW_FREERATIO);
		//imshow("direct", direct);

		//寻找图像拼接重叠的左右边界
		int rows = img1.rows;//img1 行
		int cols = img1.cols;//img1 列
		int left = 0;//左边界起始
		int right = img1.cols;//右边界起始
		Mat grayimg1, graywarpimg;
		cvtColor(img1, grayimg1, COLOR_BGR2GRAY);
		cvtColor(warpimg, graywarpimg, COLOR_BGR2GRAY);
		//左,遍历img1,warpimg的列元素，同时含有非0像素使即找到起始左边界
		for (int col = 0; col < cols; col++)
		{
			if (countNonZero(grayimg1.col(col))>0 && countNonZero(graywarpimg.col(col)) > 0)
			{
				left = col;
				break;
			}
		}
		//右A
		for (int col = cols - 1; col >= left; col--)
		{
			if (countNonZero(grayimg1.col(col)) > 0 && countNonZero(graywarpimg.col(col)) > 0)
			{
				right = col;
				break;
			}
		}
		//对重叠区域加权处理
		Mat img_res(rows,cols,CV_8UC3, cv::Scalar(0));//创捷结果存储图像
		//遍历行列处理
		for (int row = 0; row < rows; row++)
		{
			for (int col = 0; col < cols; col++)
			{
				if (img1.at<Vec3b>(row, col) == Vec3b(0, 0, 0))
					img_res.at<Vec3b>(row, col) = warpimg.at<Vec3b>(row, col);
				else if (warpimg.at<Vec3b>(row, col) == Vec3b(0, 0, 0))
					img_res.at<Vec3b>(row, col) = img1.at<Vec3b>(row, col);
				else
				{
					double srcimgLen = abs(col - left);
					double testimgLen = abs(col - right);
					double alpha = srcimgLen / (srcimgLen + testimgLen);
					img_res.at<Vec3b>(row, col) = img1.at<Vec3b>(row, col) * (1.2 - alpha) + warpimg.at<Vec3b>(row, col) * alpha;
				}
			}
		}
		Ptr<Stitcher> stitcher = Stitcher::create();
		vector<Mat> img = { img1,img2 };
		Mat img_res1;
		stitcher->stitch(img, img_res1);
		namedWindow("img_res1", WINDOW_FREERATIO);
		namedWindow("img_res", WINDOW_FREERATIO);
		imshow("img_res", img_res);
		imshow("img_res1", img_res1);
	}
	
	waitKey();
	destroyAllWindows();
	return 0;
}