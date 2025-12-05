#include "main.h"
ImageFeatures features0;
ImageFeatures features1;
int i = 1;
int T_hg()
{
	while (1)
	{			
		if (!FIFO_kp_0.empty() && !FIFO_kp_1.empty())
		{
			unique_lock<mutex> lck_kp0(kp_mtx0);
			features0 = FIFO_kp_0.front();
			FIFO_kp_0.pop();
			lck_kp0.unlock();
			unique_lock<mutex> lck_kp1(kp_mtx1);
			features1 = FIFO_kp_1.front();
			FIFO_kp_1.pop();
			lck_kp1.unlock();
			print_mtx.lock();
			cout << "match pair:" <<i++<< endl;
			print_mtx.unlock();
			vector<ImageFeatures> features = {features0 ,features1};
			(*matcher)(features, pairwise_matches);
			matcher->collectGarbage();
		}
		else if ((FIFO_kp_0.empty() || FIFO_kp_1.empty())&& kp0_flag == 1 && kp1_flag == 1) break;
		//if (kp0_flag == 1 && kp1_flag == 1) break;
	}
	print_mtx.lock();
	cout << "match end" << endl;
	print_mtx.unlock();
	return 0;
}