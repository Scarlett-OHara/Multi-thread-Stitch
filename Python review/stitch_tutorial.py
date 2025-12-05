from statistics import median

import cv2 as cv
import numpy as np
from stitching.warper import  Warper
from stitching.cropper import Cropper
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

Width,Height = 640,480
#image = cv.imread("./img/weir_1.jpg")

warper = Warper(warper_type="spherical")
warper.set_scale([camera_left,camera_medium,camera_right])

image_left =   np.full((Height,Width,3),(255,0,0),dtype=np.uint8)
image_medium = np.full((Height,Width,3),(0,255,0),dtype=np.uint8)
image_right =  np.full((Height,Width,3),(0,0,255),dtype=np.uint8)
#对图片进行投影变换
imgs = [image_left,image_medium,image_right]
imgs_sizes = [(640,480),(640,480),(640,480)]

warped_imgs = list(warper.warp_images(imgs,cameras,1))
for i in range(3):
    cv.imshow(f'img{i}',warped_imgs[i])
warped_masks = list(warper.create_and_warp_masks(imgs_sizes,cameras,1))
'''for m in range(3):
    cv.imshow(f'img_mask{m}',warped_masks[m])'''
corners,sizes = warper.warp_rois(imgs_sizes,cameras,1)

'''for n in range(3):
    print(corners[n],'   ',sizes[n],end='\n')'''
#创建裁剪器
'''cv.imshow("mask_left",warped_masks[0])
cv.imshow("mask_medium",warped_masks[1])
cv.imshow("mask_right",warped_masks[2])'''
cropper = Cropper()
mask_result = cropper.estimate_panorama_mask(warped_imgs,warped_masks,corners,sizes)
cv.imshow("mask_result",mask_result)
print(mask_result.shape)
print(corners)
#以最左图像角点为基础，对各掩码角点归一化
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

#创建画布
canvas = np.full((canvas_height,canvas_width,3),(0,0,0),dtype=np.uint8)
#cv.imshow("canvas",canvas)
#将单通道掩码转换成BGR三色，BGR 左->右
BGR_masks = []
colors = [(255,0,0),(0,255,0),(0,0,255)]
for mask,color in zip(warped_masks,colors):
    BGR_masks.append(set_mask_color(mask,color))

cv.imshow("mask0",BGR_masks[0])
cv.imshow("mask1",BGR_masks[1])
cv.imshow("mask2",BGR_masks[2])

for mask,left_up,right_down in zip(BGR_masks,left_up_corners,right_down_corners):
    canvas[left_up[1]:right_down[1],left_up[0]:right_down[0]] = canvas[left_up[1]:right_down[1],left_up[0]:right_down[0]] | mask
cv.imshow('stitch_area',canvas)
'''canvas = cv.cvtColor(canvas,cv.COLOR_BGR2GRAY)
canvas = cv.threshold(canvas,5,255,cv.THRESH_BINARY)[1]
cv.imshow('canvas_gray',canvas)'''
camera_3 = [camera_left,camera_medium,camera_right]
scale = median([cam.focal for cam in camera_3])
wp = cv.PyRotationWarper("spherical",scale)

K = camera_left.K()
K = K.astype(np.float32)
dst_pt = wp.warpPoint((0,0),K,camera_left.R)
print(dst_pt,type(dst_pt))

image_info = wp.warpRoi((640,480),K,camera_left.R)
pt = image_info[0:2]
sz = image_info[2:4]
print(pt,sz)
cv.waitKey()

