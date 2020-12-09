
'''
this file is for extracting the bounding box of STB SK set 
- before doing anything, set the camera parameters first
then
- read the annotation file which contains the 21 joints' xyz location
- transmition: transmit the xyz coordinates into uvd coordinates
- find the max and min of uv coordinates
- shift and crop and then get the [x,y,w,h] of bbox
- loop util the last image processed
'''
import os
import numpy as np 
import scipy.io as sio
import os.path as osp
import math
import numpy.linalg as LA
import torch
import cv2
import pickle

def set_cam():
    """
    to set up the camera parameters
    """
    # the color camera internal matrix
    SK_fx_color = 607.92271
    SK_fy_color = 607.88192
    SK_tx_color = 314.78337
    SK_ty_color = 236.42484

    color_K = np.zeros((3,3))
    color_K[0][0] = SK_fx_color
    color_K[1][1] = SK_fy_color
    color_K[2][2] = 1
    color_K[0][2] = SK_tx_color
    color_K[1][2] = SK_ty_color

    def SK_rot_mx(rot_vec):
        """
        use Rodrigues' rotation formula to transform the rotation vector into rotation matrix
        :param rot_vec:
        :return:
        """
        theta = LA.norm(rot_vec) # 求二范数
        vector = np.array(rot_vec) * math.sin(theta / 2.0) / theta
        a = math.cos(theta / 2.0)
        b = -vector[0]
        c = -vector[1]
        d = -vector[2]
        return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)],
                        [2 * (b * c - a * d), a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
                        [2 * (b * d + a * c), 2 * (c * d - a * b), a * a + d * d - b * b - c * c]])

    # used to transfer depth coordination to color camera coordination
    SK_rot_vec = [0.00531, -0.01196, 0.00301]
    SK_trans_vec = [-24.0381, -0.4563, -1.2326]  # mm
    SK_rot = SK_rot_mx(SK_rot_vec)

    return SK_rot_vec, SK_trans_vec, SK_rot, color_K

def uv2bbox(uv, crop_size):
    """
    get bounding box of joints
    :param uv: N x K x 2
    :return: N x 4, [x, y, w, h]
    """
    # ori_h = 480
    # ori_w = 640
    crop_size = torch.Tensor(crop_size)
    crop_size = crop_size.unsqueeze(0).expand(uv.shape[0], 2).int()
    # h, w = crop_size

    pt_max, _ = torch.max(torch.from_numpy(uv).int(), 1)  # N x 2
    pt_min, _ = torch.min(torch.from_numpy(uv).int(), 1)  # N x 2
    w_h = pt_max - pt_min + 1
    pad_wh = (crop_size - w_h)//2
    print(pad_wh[pad_wh<0])

    # return torch.cat((pt_min-pad_wh, pt_max - pt_min + 1 + (crop_size - pad_wh)), 1).int().numpy()  # N x 4
    return torch.cat((pt_min-pad_wh, crop_size), 1).int().numpy()  # N x 4

def SK_xyz_depth2color(depth_xyz, trans_vec, rot_mx):
    """
    :param depth_xyz: N x 21 x 3, trans_vec: 3, rot_mx: 3 x 3
    :return: color_xyz: N x 21 x 3
    """
    color_xyz = depth_xyz - np.tile(trans_vec, [depth_xyz.shape[0], depth_xyz.shape[1], 1])
    return color_xyz.dot(rot_mx)


def crop_pad_im_from_bounding_rect(im, bb):
    """
    :param im: H x W x C
    :param bb: x, y, w, h (may exceed the image region)
    :return: cropped image
    """
    crop_im = im[max(0, bb[1]):min(bb[1] + bb[3], im.shape[0]), max(0, bb[0]):min(bb[0] + bb[2], im.shape[1]), :]

    if bb[1] < 0:
        crop_im = cv2.copyMakeBorder(crop_im, -bb[1], 0, 0, 0,  # top, bottom, left, right, bb[3]-crop_im.shape[0]
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    if bb[1] + bb[3] > im.shape[0]:
        crop_im = cv2.copyMakeBorder(crop_im, 0, bb[1] + bb[3] - im.shape[0], 0, 0,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    if bb[0] < 0:
        crop_im = cv2.copyMakeBorder(crop_im, 0, 0, -bb[0], 0,  # top, bottom, left, right
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    if bb[0] + bb[2] > im.shape[1]:
        crop_im = cv2.copyMakeBorder(crop_im, 0, 0, 0, bb[0] + bb[2] - im.shape[1],
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    return crop_im

def get_SK_bbox(ann_path, image_dir, out_dir):
    """
    to get one subset of bbox in SK set [B1Counting, ...]
        ann_path: path to one annotation file
        image_dir: path to subset image folder
        out_dir: folder to save the bbox npy file
    """
    # set cam para
    SK_rot_vec, SK_trans_vec, SK_rot, color_K = set_cam()
    
    # read xyz annotation
    SK_depth_xyz = sio.loadmat(ann_path)["handPara"].transpose((2, 1, 0))  # N x K x 3
    SK_color_xyz = SK_xyz_depth2color(SK_depth_xyz, SK_trans_vec, SK_rot) # N x K x 3

    ## tansmit xyz to uvd
    # 第一种方案
    # SK_uv = np.zeros(SK_color_xyz.shape) # N, K, 3
    # R = np.zeros((3, 4))
    # R[0, 0] = 1
    # R[1, 1] = 1
    # R[2, 2] = 1
    # for i in range(SK_color_xyz.shape[0]):
    #     anno_uv_l = np.matmul(color_K, R) # 3, 4
    #     expand_xyz_l = np.append(SK_color_xyz[i].transpose((1,0)), np.ones((1, 21)), axis=0)
    #     anno_uv_l = np.matmul(anno_uv_l, expand_xyz_l) # 3,K
    #     depth = np.tile(anno_uv_l[2], (3,1))
    #     anno_uv_l /= depth # 3, K
    #     SK_uv[i] = anno_uv_l.transpose((1,0))
    # SK_uv = SK_uv[:,:,:2]

    # 第二种方案
    SK_uv = np.zeros(SK_color_xyz.shape)
    for i in range(SK_color_xyz.shape[0]):
        SK_uv[i] = np.matmul(color_K, SK_color_xyz[i].T).T
    SK_d = SK_uv[:, :, -1:]
    SK_uv = SK_uv[:,:, :2] / SK_uv[:, :, -1:]

    # test uv2color_xyz
    SK_uvd = np.concatenate((SK_uv, SK_d), axis=2)
    # test the color uv to bbox uv
    # K = np.identity(3)
    # K[0][2] = 50
    # K[1][2] = 50
    # test_uvd = SK_uvd[0].copy()
    # test_uvd[:,-1] = 1
    # test_shift_uv = np.matmul(K, test_uvd.T).T

    # uvd2color_xyz(SK_uvd[0], color_K)

    # get bbox
    bbox = uv2bbox(SK_uv, (256,256))

    # test draw
    # img = cv2.imread(osp.join(image_dir, 'SK_color_1.png'))
    # draw_joints(SK_uv[1], bbox[1], img)

    # save bbox
    file_name = ann_path.split('/')[-1][:-4]
    np.save(osp.join(out_dir, '{}.npy'.format(file_name)), bbox)
    return bbox, SK_uvd, SK_color_xyz, color_K

def uvd2color_xyz(uvd, color_K):
    """
    to recover 3d coords in camera coordinate from the uvd 
        uvd: uv and depth coords of one frame (21, 3)
        color_K: the internal matrix of the color camera
    return
        coor3d: the camera 3d coords (21, 3)
    """
    print(uvd.shape)
    def to_torch(ndarray):
        if type(ndarray).__module__ == 'numpy':
            return torch.from_numpy(ndarray).float()
        elif not torch.is_tensor(ndarray):
            raise ValueError("Cannot convert {} to torch tensor"
                            .format(type(ndarray)))
        return ndarray.float()

    matrix = np.linalg.inv(color_K) #take the inversion of matrix

    coor3d = uvd.copy()
    coor3d[:,:2] *= coor3d[:, 2:]
    coor3d = torch.matmul(to_torch(coor3d), to_torch(matrix).transpose(0, 1))

    return coor3d

def draw_joints(uv, bbox, img):
    '''
    to draw the joints and bbox of one frame
        uv: 21 joints 2d coordinates (21, 2)
        bbox: the bbox of the frame [x,y,w,h]
        img: initial image read by cv2 [480, 640]
    '''
    print(bbox)
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    # draw joints
    for point in uv:
        cv2.circle(img, (int(point[0]),int(point[1])), point_size, point_color, thickness)
    
    # draw bbox
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 4)
    cv2.imwrite('./testout.png', img)

if __name__ == "__main__":
    # ann_path = '../STB/B1Counting_SK.mat'
    # image_dir = '../STB/images'
    # out_dir = '../STB/test_out_bbox'
    # get_SK_bbox(ann_path, image_dir, out_dir)

    # root_dir = '../STB/STB'
    root_dir = '../hand-pose-STB'
    subsets = ['B1Counting', 'B2Counting','B3Counting', 'B4Counting','B5Counting', 'B6Counting']
    f = open(osp.join(root_dir, 'STB_SK.pickle'), 'wb')
    to_pickle = {}
    for subset in subsets:
        print('processing subset:{}'.format(subset))
        ann_path = osp.join(root_dir, 'labels', subset + '_SK.mat')
        # image_dir = osp.join(root_dir, 'images', subset + '_SK')
        image_dir = osp.join(root_dir, 'images', subset)
        bbox_out_dir = '../hand-pose-STB/hand_bbox'
        cropped_out_dir = osp.join('../hand-pose-STB/cropped_images', subset + '_SK')

        bbox, SK_uvd, SK_color_xyz, color_K = get_SK_bbox(ann_path, image_dir, bbox_out_dir) # (1500, 4)
        matrix = np.tile(np.expand_dims(color_K, axis=0), (1500,1,1)) # internal para for color_xyz-->bbox uvd
        matrix[:,0,2] = matrix[:,0,2] - bbox[:, 0]
        matrix[:,1,2] = matrix[:,1,2] - bbox[:, 1]

        subset_pickle = {'sk':
                            {'coor3d':SK_color_xyz,
                             'coor2d':SK_uvd,
                             'matrix':matrix}
                        }
        to_pickle[subset] = subset_pickle

        for ix, bb in enumerate(bbox):
        # for ix in range(3):
            bb = bbox[ix]
            img_path = osp.join(image_dir, 'SK_color_{}.png'.format(ix))
            img = cv2.imread(img_path)
            crop = crop_pad_im_from_bounding_rect(img, bb)
            cv2.imwrite(osp.join(cropped_out_dir, 'SK_color_{}.png'.format(ix)), crop)

    pickle.dump(to_pickle, f)
    f.close()

    f = open(osp.join(root_dir, 'STB_SK.pickle'), 'rb')
    db = pickle.load(f)
    print(db['B1Counting']['sk']['matrix'][0])
    f.close()


