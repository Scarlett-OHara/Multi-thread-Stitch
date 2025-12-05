import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

class stitch:
    pt_bas:tuple
    scale:float

    def __init__(self,camera:cv.detail.CameraParams,color:tuple,size:tuple,type:str):
        self.camera = camera
        self.color = color
        self.type = type
        self.size = size
        self.K_CV32F = None
        self.warper:cv.PyRotationWarper
        self.image = np.full((self.size[1],self.size[0],3),self.color,dtype=np.uint8)
        self.warped_image = None
        self.warped_pos:tuple
        self.warped_size:tuple
        self.abs_pos:tuple
        self.camera_flag = False

    #set camera params
    def camera_init(self,focal:float,aspect:float,ppx:float,ppy:float,R:cv.typing.MatLike,t:cv.typing.MatLike):
        self.camera.focal = focal
        self.camera.aspect = aspect
        self.camera.ppx = ppx
        self.camera.ppy = ppy
        self.camera.R = R
        self.camera.t = t
        self.K_CV32F = self.camera.K()
        self.K_CV32F = self.K_CV32F.astype(np.float32)
        self.camera_flag = True

    #print camera params
    def camera_info_print(self):
        if self.camera_flag:
            print(f'内参:\n{self.camera.K()}',end='\n')
            print(f'旋转:\n{self.camera.R}',  end='\n')
            print(f'平移:\n{self.camera.t}',  end='\n')
        else:
            raise ValueError("Camera not init")

    #get orginal image
    def get_image(self):
        return self.image

    #init warper params
    def warper_init(self):
        self.warper = cv.PyRotationWarper(self.type,stitch.scale)

    #warp image and get warp result
    def warp_image(self):
        if self.camera_flag:
            self.warped_image = self.warper.warp(self.image,self.K_CV32F,self.camera.R,cv.INTER_LINEAR,cv.BORDER_CONSTANT)[1]
            image_info = self.warper.warpRoi(self.size,self.K_CV32F,self.camera.R)
            self.warped_pos = image_info[0:2]
            self.warped_size = image_info[2:4]
        else:
            raise ValueError("Camera not init")

        return self.warped_image

    #get warped_images:absolute location on canvas
    def set_abs_pos(self):
        canvas_cor_leftup = self.map_pts_wp2cv((0, 0))
        canvas_cor_rightdw = tuple(m + n-1 for m, n in zip(canvas_cor_leftup, self.warped_size))
        self.abs_pos = (canvas_cor_leftup,canvas_cor_rightdw)
        return self.abs_pos

    #map warped image to canvas
    def map_img_wp2cv(self,Canvas):
        tl,br = self.abs_pos
        Canvas[tl[1]:br[1]+1,tl[0]:br[0]+1] = Canvas[tl[1]:br[1]+1,tl[0]:br[0]+1] | self.warped_image
        return Canvas

    # map points (or a point) from orign image to warped image
    def map_pts_org2wp(self,pt_org:tuple):
        if (0<=pt_org[0]<self.size[0]-1) & (0<=pt_org[1]<self.size[1]-1):
            pts_tmp = self.warper.warpPoint(pt_org,self.K_CV32F,self.camera.R)
        else:
            raise ValueError('坐标超出图像范围')
        pts_dst = (int(pts_tmp[0]-self.warped_pos[0]),int(pts_tmp[1]-self.warped_pos[1]))
        return pts_dst

    # map a points (or a point) from warped image to orign image
    def map_pts_wp2org(self,pt_org:tuple):
        pts_dst = self.warper.warpPointBackward(pt_org, self.K_CV32F, self.camera.R)
        return pts_dst

    # map a points (or a point) from warped image to canvas
    def map_pts_wp2cv(self,pt_org:tuple):
        if (0<=pt_org[0]<=self.size[0]) & (0<=pt_org[1]<=self.size[1]):
            pts_dst = (pt_org[0]+self.warped_pos[0]-stitch.pt_bas[0],pt_org[1]+self.warped_pos[1]-stitch.pt_bas[1])
        else:
            raise ValueError('坐标超出图像范围')
        return pts_dst

    # map a points (or a point) from canvas to warped image
    def map_pts_cv2wp(self,pt_org:tuple):
        tl, br = self.abs_pos
        if (tl[0]<=pt_org[0]<=br[0]) & (tl[1]<=pt_org[1]<=br[1]):
            pts_dst = (pt_org[0]-(self.warped_pos[0]-stitch.pt_bas[0]),pt_org[1]-(self.warped_pos[1]-stitch.pt_bas[1]))
        else:
            raise ValueError("坐标超出当前图像在画布中的范围")
        return pts_dst



#read yaml
lm = cv.FileStorage("./stereo_calib-lm.yaml",cv.FILE_STORAGE_READ)#左中相机参数
mr = cv.FileStorage("./stereo_calib-mr.yaml",cv.FILE_STORAGE_READ)#中右相机参数

#set camera
camera_left = cv.detail.CameraParams()
camera_medium = cv.detail.CameraParams()
camera_right = cv.detail.CameraParams()

#set stitch class
Left = stitch(camera_left,(255,0,0),(640,480),"spherical")
Medium = stitch(camera_medium,(0,255,0),(640,480),"spherical")
Right = stitch(camera_right,(0,0,255),(640,480),"spherical")

#Left.camera_info_print()
#Left.warp_image()

#set camera param
print('----------初始化左边相机开始----------',end='\n')
Left.camera_init(np.float32 (lm.getNode("camera_matrix_left").mat()[0][0]),
                 np.float32(lm.getNode("camera_matrix_left").mat()[1][1])/np.float32 (lm.getNode("camera_matrix_left").mat()[0][0]),
                 np.float32 (lm.getNode("camera_matrix_left").mat()[0][2]),np.float32 (lm.getNode("camera_matrix_left").mat()[1][2]),
                 np.float32(np.linalg.inv(lm.getNode("R").mat())),np.float32(lm.getNode("T").mat()))
Left.camera_info_print()
print('----------初始化左边相机结束----------',end='\n')


print('----------初始化中间相机开始----------',end='\n')
medium_K = np.float32((lm.getNode("camera_matrix_right").mat()+mr.getNode("camera_matrix_left").mat())/2) #两文件中中间相机内参不同，求平均

Medium.camera_init(medium_K[0][0],medium_K[1][1]/medium_K[0][0],
                   medium_K[0][2],medium_K[1][2],
                   np.eye(3,3,dtype=np.float32),np.zeros((3,1),dtype=np.float32))
Medium.camera_info_print()
print('----------初始化中间相机结束----------',end='\n')

print('----------初始化右边相机开始----------',end='\n')
Right.camera_init(np.float32 (mr.getNode("camera_matrix_right").mat()[0][0]),
                   np.float32(mr.getNode("camera_matrix_right").mat()[1][1])/np.float32 (mr.getNode("camera_matrix_right").mat()[0][0]),
                   np.float32 (mr.getNode("camera_matrix_right").mat()[0][2]),np.float32 (mr.getNode("camera_matrix_right").mat()[1][2]),
                   np.float32(mr.getNode("R").mat()),np.float32(mr.getNode("T").mat()))
Right.camera_info_print()
print('----------初始化右边相机结束----------',end='\n')

#求warper缩放系数,所有相机的焦距中位数
scale = median([Left.camera.focal,Medium.camera.focal,Right.camera.focal])
stitch.scale = scale

#对warper进行初始化
Left.warper_init()
Medium.warper_init()
Right.warper_init()

#得到RGB原图像
image_left   = Left.get_image()
image_medium = Medium.get_image()
image_right  = Right.get_image()
'''cv.imshow("image_left",image_left)
cv.imshow("image_medium",image_medium)
cv.imshow("image_right",image_right)'''

#得到投影变换后图像
warped_left   = Left.warp_image()
warped_medium = Medium.warp_image()
warped_right  = Right.warp_image()
print(f'左投影图尺寸:{Left.warped_size}')
print(f'中投影图尺寸:{Medium.warped_size}')
print(f'右投影图尺寸:{Right.warped_size}')
print()
'''cv.imshow("warped_left",warped_left)
cv.imshow("warped_medium",warped_medium)
cv.imshow("warped_right",warped_right)'''

#得到各图像的相对位置坐标
cor_left = Left.warped_pos
cor_medium = Medium.warped_pos
cor_right = Right.warped_pos
print(f'左图相对坐标:{cor_left}')
print(f'中图相对坐标:{cor_medium}')
print(f'右图相对坐标:{cor_right}')
print()

#设置基准坐标
x,y = zip(cor_left,cor_medium,cor_right)
stitch.pt_bas = (min(x),min(y))
print(f'基准坐标:{stitch.pt_bas}')
print()

#设置并得到投影图像在画布中绝对位置
left_abs_pos =   Left.set_abs_pos()
medium_abs_pos = Medium.set_abs_pos()
right_abs_pos =  Right.set_abs_pos()
print(f'左图绝对位置{left_abs_pos[0]},{left_abs_pos[1]}')
print(f'中图绝对位置{medium_abs_pos[0]},{medium_abs_pos[1]}')
print(f'右图绝对位置{right_abs_pos[0]},{right_abs_pos[1]}')
print()

#根据右下点位置得到拼接画布范围
x,y = zip(left_abs_pos[1],medium_abs_pos[1],right_abs_pos[1])
area = (max(x),max(y))
print(f'画布Width:{area[0]},Height:{area[1]}')

#创建拼接画布
Canvas = np.full((area[1]+1,area[0]+1,3),(0,0,0),dtype=np.uint8)

#根据绝对位置将投影图映射至画布
Canvas = Left.map_img_wp2cv(Canvas)
Canvas = Medium.map_img_wp2cv(Canvas)
Canvas = Right.map_img_wp2cv(Canvas)

#显示拼接结果
fig = plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1))
ax3 = plt.subplot2grid((3, 3), (0, 2))
ax4 = plt.subplot2grid((3, 3), (1, 0))
ax5 = plt.subplot2grid((3, 3), (1, 1))
ax6 = plt.subplot2grid((3, 3), (1, 2))
ax7 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

ax1.imshow(cv.cvtColor(image_left,cv.COLOR_BGR2RGB))
ax2.imshow(cv.cvtColor(image_medium,cv.COLOR_BGR2RGB))
ax3.imshow(cv.cvtColor(image_right,cv.COLOR_BGR2RGB))
ax4.imshow(cv.cvtColor(warped_left,cv.COLOR_BGR2RGB))
ax5.imshow(cv.cvtColor(warped_medium,cv.COLOR_BGR2RGB))
ax6.imshow(cv.cvtColor(warped_right,cv.COLOR_BGR2RGB))
ax7.imshow(cv.cvtColor(Canvas,cv.COLOR_BGR2RGB))

plt.show()

print('----------测试点映射----------')
print('从源图像映射至投影图')
print(Left.map_pts_org2wp((0,0)),Medium.map_pts_org2wp((0,0)),Right.map_pts_org2wp((0,0)))
print('从投影图映射至画布')
print(Left.map_pts_wp2cv((0,0)),Medium.map_pts_wp2cv((0,0)),Right.map_pts_wp2cv((0,0)))
print('从画布映射至投影图')
print(Left.map_pts_cv2wp((0,0)),Medium.map_pts_cv2wp((229, 44)),Right.map_pts_cv2wp((427, 54)))

print(Left.map_pts_wp2org((1,66)))

