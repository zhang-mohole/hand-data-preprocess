import numpy as np 
import os.path as osp
import scipy.io as sio
import cv2
import torch
import pickle

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

def anno2bbox(anno_path, out_dir):
    '''
    * anno_path: BB subset annotation mat file 
    * image_dir: path to subset image folder
    * out_dir: folder to save the bbox npy file
    * out: bbox, 3d position(xyz), 2d position(uv)
    here we only consider the *_left image
    '''
    hand_para = sio.loadmat(anno_path)['handPara']

    R_l, R_r, K = set_cam()

    uv_l = np.zeros((1500, 21, 2))
    xyz_l = hand_para.transpose((2, 1, 0)) # (1500, 21, 3)
    # uv_r = np.zeros((1500, 21, 2))
    for im_id in range(1500):
        anno_xyz_l = hand_para[:, :, im_id] # 3, K

        anno_uv_l = np.matmul(K, R_l) # 3, 4
        expand_xyz_l = np.append(anno_xyz_l, np.ones((1, 21)), axis=0)
        anno_uv_l = np.matmul(anno_uv_l, expand_xyz_l) # 3,K

        scale = np.tile(anno_uv_l[2], (3,1))
        anno_uv_l /= scale

        uv_l[im_id] = anno_uv_l.transpose((1,0))[:,:2] # K, 2   
    
    bbox_l = uv2bbox(uv_l, (256,256))

    # save bbox
    file_name = ann_path.split('/')[-1][:-4]
    np.save(osp.join(out_dir, '{}.npy'.format(file_name)), bbox)
    return bbox_l, xyz_l, uv_l


def draw_joints(uv, img):
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    
    for point in uv:
        cv2.circle(img, (int(point[0]),int(point[1])), point_size, point_color, thickness)
    cv2.imwrite('./testout.png', img)

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

if __name__ == "__main__":
    root_dir = '../hand-pose-STB'
    subsets = ['B1Counting', 'B2Counting','B3Counting', 'B4Counting','B5Counting', 'B6Counting',
               'B1Random', 'B2Random','B3Random', 'B4Random','B5Random', 'B6Random']
    f = open(osp.join(root_dir, 'STB_BB.pickle'), 'wb')
    to_pickle = {}
    for subset in subsets:
        print('processing subset:{}'.format(subset))
        ann_path = osp.join(root_dir, 'labels', subset + '_BB.mat')
        image_dir = osp.join(root_dir, 'images', subset)
        bbox_out_dir = '../hand-pose-STB/hand_bbox'
        cropped_out_dir = osp.join('../hand-pose-STB/cropped_images', subset + '_BB')

        bbox, BB_xyz, BB_uv = anno2bbox(ann_path, bbox_out_dir)

        subset_pickle = {'bb':
                            {'coor3d':BB_xyz,
                             'coor2d':BB_uv}
                        }
        to_pickle[subset] = subset_pickle

        for ix, bb in enumerate(bbox):
            bb = bbox[ix]
            img_path = osp.join(image_dir, 'BB_left_{}.png'.format(ix))
            img = cv2.imread(img_path)
            crop = crop_pad_im_from_bounding_rect(img, bb)
            cv2.imwrite(osp.join(cropped_out_dir, 'BB_left_{}.png'.format(ix)), crop)

    pickle.dump(to_pickle, f)
    f.close()

    f = open(osp.join(root_dir, 'STB_BB.pickle'), 'rb')
    db = pickle.load(f)
    print(db['B1Counting']['bb']['coor2d'][0])
    f.close()
