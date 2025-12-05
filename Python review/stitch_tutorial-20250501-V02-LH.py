# %% import module
import cv2 as cv
import numpy as np
from stitching.warper import  Warper
from stitching.cropper import Cropper
import matplotlib.pyplot as plt
from statistics import median
# %% subfunctions
def camera_init(camera:cv.detail.CameraParams,focal:float,aspect:float,ppx:float,ppy:float,R:cv.typing.MatLike,t:cv.typing.MatLike):
    camera.focal = focal
    camera.aspect = aspect
    camera.ppx = ppx
    camera.ppy = ppy
    camera.R = R
    camera.t = t

def print_param(camera:cv.detail.CameraParams,name:str):
    print('---'+name+'---',end='\n')
    print('K:\n', camera.K(),end='\n')
    print('R:\n', camera.R,  end='\n')
    print('t:\n', camera.t,  end='\n')

def set_mask_color(mask,color:tuple):
    mask = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)
    full_color = np.full(mask.shape,color,dtype=np.uint8)
    color_mask = mask & full_color
    return color_mask

# map points (or a point) from orign image to warped image
def map_pts_org2wp(pts_org:tuple, camera:cv.detail.CameraParams,scale):
    wp = cv.PyRotationWarper("spherical",scale)
    K = camera.K()
    K = K.astype(np.float32)
    #获得相对坐标
    pts_dst_tmp = wp.warpPoint(pts_org,K,camera.R)
    #获得图片大小及角点相对坐标
    pts_cor = wp.warpRoi((640,480),K,camera.R)
    #得到绝对坐标
    pts_dst = []
    for p in zip(pts_dst_tmp,(pts_cor[0],pts_cor[1])):
        pts_dst.append(int(p[0]-p[1]))
    return tuple(pts_dst)

# map a points (or a point) from warped image to orign image
def map_pts_wp2org(pt_org, camera):
    ''' and code here '''
    return pts_dst

# map a points (or a point) from warped image to canvas
def map_pts_wp2cv(pt_org:tuple, camera:cv.detail.CameraParams,scale):
    ''' and code here '''
    wp = cv.PyRotationWarper("spherical", scale)
    K = camera.K()
    K = K.astype(np.float32)
    pts_cor = wp.warpRoi((640,480),K,camera.R)
    pts_dst = (pt_org[0]+pts_cor[0]+481,pt_org[1]+pts_cor[1]-224)
    return pts_dst

# map a points (or a point) from canvas to warped image
def map_pts_cv2wp(pt_org:tuple, camera:cv.detail.CameraParams,scale):
    ''' and code here '''
    wp = cv.PyRotationWarper("spherical", scale)
    K = camera.K()
    K = K.astype(np.float32)
    pts_cor = wp.warpRoi((640, 480), K, camera.R)
    pts_dst = (pt_org[0]-(pts_cor[0]+481),pt_org[1]-(pts_cor[1]-224))
    return pts_dst

# %% load file, and get paramters of cameras

Width,Height = 640,480
#image = cv.imread("./img/weir_1.jpg")

#左中相机参数
lm = cv.FileStorage("./stereo_calib-lm.yaml",cv.FILE_STORAGE_READ)
mr = cv.FileStorage("./stereo_calib-mr.yaml",cv.FILE_STORAGE_READ)
camera_left = cv.detail.CameraParams()
camera_init(camera_left,np.float32 (lm.getNode("camera_matrix_left").mat()[0][0]),
            np.float32(lm.getNode("camera_matrix_left").mat()[1][1])/np.float32 (lm.getNode("camera_matrix_left").mat()[0][0]),
            np.float32 (lm.getNode("camera_matrix_left").mat()[0][2]),np.float32 (lm.getNode("camera_matrix_left").mat()[1][2]),
            np.float32(np.linalg.inv(lm.getNode("R").mat())),np.float32(lm.getNode("T").mat()))
print_param(camera_left,"camera_left")

camera_right = cv.detail.CameraParams()
camera_init(camera_right,np.float32 (mr.getNode("camera_matrix_right").mat()[0][0]),
            np.float32(mr.getNode("camera_matrix_right").mat()[1][1])/np.float32 (mr.getNode("camera_matrix_right").mat()[0][0]),
            np.float32 (mr.getNode("camera_matrix_right").mat()[0][2]),np.float32 (mr.getNode("camera_matrix_right").mat()[1][2]),
            np.float32(mr.getNode("R").mat()),np.float32(mr.getNode("T").mat()))
print_param(camera_right,"camera_right")

medium_K = np.float32((lm.getNode("camera_matrix_right").mat()+mr.getNode("camera_matrix_left").mat())/2) #对中间相机的两个内参矩阵求平均
camera_medium = cv.detail.CameraParams()
camera_init(camera_medium,medium_K[0][0],medium_K[1][1]/medium_K[0][0],
            medium_K[0][2],medium_K[1][2],
            np.eye(3,3,dtype=np.float32),np.zeros((3,1),dtype=np.float32))
print_param(camera_medium,"camera_medium")
#cv.detail.waveCorrect()
cameras = [camera_left,camera_medium,camera_right]
scale = median([cam.focal for cam in cameras])
# %% warpe images, masks, and get coresponding corners as well as sizes
warper = Warper(warper_type="spherical")
warper.set_scale([camera_left,camera_medium,camera_right])

image_left =   np.full((Height,Width,3),(255,0,0),dtype=np.uint8)
image_medium = np.full((Height,Width,3),(0,255,0),dtype=np.uint8)
image_right =  np.full((Height,Width,3),(0,0,255),dtype=np.uint8)
#对图片进行投影变换
imgs = [image_left,image_medium,image_right]
imgs_sizes = [(Width,Height),(Width,Height),(Width,Height)]

warped_imgs = list(warper.warp_images(imgs,cameras,1))
'''for i in range(3):
    cv.imshow(f'img{i}',warped_imgs[i])'''
warped_masks = list(warper.create_and_warp_masks(imgs_sizes,cameras,1))
'''for m in range(3):
    cv.imshow(f'img_mask{m}',warped_masks[m])'''
corners,sizes = warper.warp_rois(imgs_sizes,cameras,1)

for n in range(3):
    print(corners[n],'   ',sizes[n],end='\n')

# %% cropper images and show stitched mask
#创建裁剪器
cropper = Cropper()
mask_result = cropper.estimate_panorama_mask(warped_imgs,warped_masks,corners,sizes)
cv.imshow("mask_result",mask_result)
#print(mask_result.shape)

# %% calculate size and position of canvas and each image
#以最左图像角点为基础，对各掩码角点归一化
# print('所有ROI角点坐标:',corners)
# print('所有ROI尺寸:',sizes)
corner_left = corners[0]
for i in range(len(corners)):
    corners[i] = tuple(a-b for a,b in zip(corners[i],corner_left))
#print(corners)
#左上角坐标
left_up_corners = corners
print('左上角坐标:',left_up_corners)
#根据归一化后角点及其尺寸计算画布大小
widthes = []
heights = []
right_down_corners = []
for cr,sz in zip(corners, sizes):
    widthes.append(cr[0]+sz[0])
    heights.append(cr[1]+sz[1])
    right_down_corners.append((cr[0]+sz[0],cr[1]+sz[1]))
canvas_width = max(widthes)
canvas_height= max(heights)
print('右下角坐标:',right_down_corners)
print(f'画布 width:{canvas_width},height:{canvas_height}')

# %% plot figure
#创建画布
canvas = np.full((canvas_height,canvas_width,3),(0,0,0),dtype=np.uint8)
#cv.imshow("canvas",canvas)
#将单通道掩码转换成BGR三色，BGR 左->右
BGR_masks = []
colors = [(255,0,0),(0,255,0),(0,0,255)]
for mask,color in zip(warped_masks,colors):
    BGR_masks.append(set_mask_color(mask,color))

for mask,left_up,right_down in zip(BGR_masks,left_up_corners,right_down_corners):
    canvas[left_up[1]:right_down[1],left_up[0]:right_down[0]] = canvas[left_up[1]:right_down[1],left_up[0]:right_down[0]] | mask

'''cv.imshow("image left",image_left)
cv.imshow("image middle",image_medium)
cv.imshow("image right",image_right)
cv.imshow("mask left",BGR_masks[0])
cv.imshow("mask middle",BGR_masks[1])
cv.imshow("mask right",BGR_masks[2])
cv.imshow('stitch_area',canvas)'''

fig = plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1))
ax3 = plt.subplot2grid((3, 3), (0, 2))
ax4 = plt.subplot2grid((3, 3), (1, 0))
ax5 = plt.subplot2grid((3, 3), (1, 1))
ax6 = plt.subplot2grid((3, 3), (1, 2))
ax7 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
ax1.imshow(cv.cvtColor(image_left, cv.COLOR_BGR2RGB))
ax2.imshow(cv.cvtColor(image_medium, cv.COLOR_BGR2RGB))
ax3.imshow(cv.cvtColor(image_right, cv.COLOR_BGR2RGB))
ax4.imshow(cv.cvtColor(BGR_masks[0], cv.COLOR_BGR2RGB))
ax5.imshow(cv.cvtColor(BGR_masks[1], cv.COLOR_BGR2RGB))
ax6.imshow(cv.cvtColor(BGR_masks[2], cv.COLOR_BGR2RGB))
ax7.imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
print(map_pts_org2wp((0,0),camera_left,scale))
print(map_pts_org2wp((0,0),camera_medium,scale))
print(map_pts_org2wp((0,0),camera_right,scale))
print(map_pts_wp2cv((536, 419),camera_left,scale))
print(map_pts_wp2cv((0,0),camera_medium,scale))
print(map_pts_wp2cv((0,0),camera_right,scale))
print(map_pts_cv2wp((536, 419),camera_left,scale))
print(map_pts_cv2wp((705, 439),camera_medium,scale))
print(map_pts_cv2wp((924, 457),camera_right,scale))
plt.show()

cv.waitKey()


# %%