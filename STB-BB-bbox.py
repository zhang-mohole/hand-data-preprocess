import numpy as np 
import os.path as osp
import scipy.io as sio
import cv2

def set_cam():
    fx = 822.79041
    fy = 822.79041
    tx = 318.47345
    ty = 250.31296
    base = 120.054

    R_l = np.zeros((3, 4))
    R_l[0, 0] = 1
    R_l[1, 1] = 1
    R_l[2, 2] = 1
    R_r = R_l.copy()
    R_r[0, 3] = -base
    print(R_l, R_r)

    K = np.diag([fx, fy, 1.0])
    K[0, 2] = tx
    K[1, 2] = ty

    return R_l, R_r, K

def anno2bbox(anno_path):
    hand_para = sio.loadmat(anno_path)['handPara']

    R_l, R_r, K = set_cam()

    uv_l = np.zeros((1500, 21, 2))
    uv_r = np.zeros((1500, 21, 2))
    for im_id in range(1500):
        anno_xyz_l = hand_para[:, :, im_id] # 3, K

        anno_uv_l = np.matmul(K, R_l) # 3, 4
        expand_xyz_l = np.append(anno_xyz_l, np.ones((1, 21)), axis=0)
        anno_uv_l = np.matmul(anno_uv_l, expand_xyz_l) # 3,K

        scale = np.tile(anno_uv_l[2], (3,1))
        anno_uv_l /= scale
        # for k in range(21):
        #     # anno_uv_l[:, k] = anno_uv_l[:, k] / anno_uv_l(2, k)
        #     scale = anno_uv_l(2, k)
        #     anno_uv_l[:, k] /= scale

        anno_xyz_r = np.matmul(R_r, np.append(anno_xyz_l, np.ones((1,21)), axis=0)) # 3, K
        anno_uv_r = np.matmul(K, anno_xyz_r) # 3, K

        scale = np.tile(anno_uv_r[2], (3,1))
        anno_uv_r /= scale
        # for k in range(21):
        #     anno_uv_r[:, k] = anno_uv_r[:, k] / anno_uv_r(2, k)
        
        uv_l[im_id] = anno_uv_l.transpose((1,0))[:,:2]
        uv_r[im_id] = anno_uv_r.transpose((1,0))[:,:2] # K, 2
    np.save('test_uv_l.npy', uv_l)
    np.save('test_uv_r.npy', uv_r)

    img = cv2.imread('../STB/images/BB_left_0.png')
    draw_joints(uv_l[0], img)

def draw_joints(uv, img):
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    
    for point in uv:
        cv2.circle(img, (int(point[0]),int(point[1])), point_size, point_color, thickness)
    cv2.imwrite('./testout.png', img)

if __name__ == "__main__":
    # img = '../STB/images/BB_left_0.png'
    ann_path = '../STB/B1Counting_BB.mat'
    anno2bbox(ann_path)
