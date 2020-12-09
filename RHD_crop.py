from __future__ import print_function, unicode_literals

import pickle
import os
import os.path as osp
import scipy.misc
# import struct
import numpy as np
import cv2
from tqdm import tqdm


def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

def crop_pad_im_from_bounding_rect(im, bb, in_type='image'):
    """
    :param im: H x W x C
    :param bb: x, y, w, h (may exceed the image region)
    :return: cropped image
    """
    if in_type =='image':
        pad_value = (0,0,0,0)
    else:
        pad_value = (5,5,5,5)

    bb = bb.astype(np.int)
    crop_im = im[max(0, bb[1]):min(bb[1] + bb[3], im.shape[0]), max(0, bb[0]):min(bb[0] + bb[2], im.shape[1]), :]

    if bb[1] < 0:
        print('top_left point exceed image --- top')
        crop_im = cv2.copyMakeBorder(crop_im, -bb[1], 0, 0, 0,  # top, bottom, left, right, bb[3]-crop_im.shape[0]
                                     borderType=cv2.BORDER_CONSTANT, value=pad_value)
    if bb[1] + bb[3] > im.shape[0]:
        crop_im = cv2.copyMakeBorder(crop_im, 0, bb[1] + bb[3] - im.shape[0], 0, 0,
                                     borderType=cv2.BORDER_CONSTANT, value=pad_value)

    if bb[0] < 0:
        print('top_left point exceed image --- left')
        crop_im = cv2.copyMakeBorder(crop_im, 0, 0, -bb[0], 0,  # top, bottom, left, right
                                     borderType=cv2.BORDER_CONSTANT, value=pad_value)
    if bb[0] + bb[2] > im.shape[1]:
        crop_im = cv2.copyMakeBorder(crop_im, 0, 0, 0, bb[0] + bb[2] - im.shape[1],
                                     borderType=cv2.BORDER_CONSTANT, value=pad_value)
    return crop_im

def get_crop_bbox(kp_uv, kp_visible, factor=2.5, max_crop_size=256):
    if len(np.where(kp_visible)[0]) < 2:
        return None
    crop_center = kp_uv[12, :]
    if crop_center[0] < 0 or crop_center[1] > 320 or \
       crop_center[1] < 0 or crop_center[1] > 320:
        return None
    
    kp_coord_h = kp_uv[kp_visible, 1].reshape(1, -1)
    kp_coord_w = kp_uv[kp_visible, 0].reshape(1, -1)
    kp_coord_wh = np.concatenate((kp_coord_w, kp_coord_h), axis=0)

    min_coord = np.min(kp_coord_wh, axis=1)
    max_coord = np.max(kp_coord_wh, axis=1)

    crop_size = factor * np.max(np.append(max_coord - crop_center, crop_center - min_coord))
    crop_size = min(crop_size, max_crop_size)

    top_left = crop_center - crop_size/2
    top_left = np.where(top_left<0, 0, top_left) # make sure that top_left point inside the image

    bbox = np.append(top_left, np.array([crop_size, crop_size])) # top_left x,y, crop w,h

    # filter the unresonable crop
    over_right = bbox[0] + bbox[2] - 320
    over_down = bbox[1] + bbox[3] - 320
    if over_right > crop_size/3 or over_down > crop_size/3:
        return None

    return bbox

if __name__ == "__main__":
    '''
    
    '''
    # path_to_db = '/home/tlh-lxy/zmh/data/RHD_published_v2/'
    # out_dir = '/home/tlh-lxy/zmh/data/RHD_published_v2/processed'
    path_to_db = '/home/zmh/datasets/dataset/RHD_published_v2/'
    out_dir = '/home/zmh/datasets/dataset/RHD_published_v2/processed'
    # which_set = 'training'
    which_set = 'evaluation'
    f = open(osp.join(out_dir, which_set, 'RHD_%s.pickle' % which_set), 'wb')
    to_pickle = {}
    target_size = (256, 256)
    depth_target_size = (128, 128)

    ## first to read the annotation file
    f_anno = open(osp.join(path_to_db, which_set, 'anno_%s.pickle' % which_set), 'rb')
    anno_all = pickle.load(f_anno) # dict
    num_samples = len(anno_all.items())

    ## load source color image and depth image
    for sample_id, anno in tqdm(anno_all.items()):
        # load data
        # image = scipy.misc.imread(osp.join(path_to_db, which_set, 'color', '%.5d.png' % sample_id))
        image = cv2.imread(osp.join(path_to_db, which_set, 'color', '%.5d.png' % sample_id))
        mask = scipy.misc.imread(osp.join(path_to_db, which_set, 'mask', '%.5d.png' % sample_id))
        depth = scipy.misc.imread(osp.join(path_to_db, which_set, 'depth', '%.5d.png' % sample_id))
        # process rgb coded depth into float: top bits are stored in red, bottom in green channel
        depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])  # depth in meters from the camera

        # get info from annotation dictionary
        kp_coord_uv = np.array(anno['uv_vis'][:, :2])  # u, v coordinates of 42 hand keypoints, pixel
        left_kp_uv = kp_coord_uv[:21]
        right_kp_uv = kp_coord_uv[21:]
        kp_visible = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
        left_visible = kp_visible[:21]
        right_visible = kp_visible[21:]
        kp_coord_xyz = np.array(anno['xyz'])  # x, y, z coordinates of the keypoints, in meters
        left_xyz = kp_coord_xyz[:21]
        right_xyz = kp_coord_xyz[21:]
        camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

        ## get the crop bbox according to the visible keypoints
        left_bbox = get_crop_bbox(left_kp_uv, left_visible)
        right_bbox = get_crop_bbox(right_kp_uv, right_visible)

        if left_bbox is not None:
            ## crop, resize and save two hands
            left_crop = crop_pad_im_from_bounding_rect(image, left_bbox)
            left_crop = cv2.resize(left_crop, target_size, interpolation=cv2.INTER_LINEAR) #双线性插值
            cv2.imwrite(osp.join(out_dir, which_set, 'crop', '%.5d_left.png' % sample_id), left_crop)
        
            ## crop, resize and save depth map for two hands
            left_depth_crop = crop_pad_im_from_bounding_rect(depth.reshape(depth.shape+(1,)), left_bbox)
            left_depth_crop = cv2.resize(left_depth_crop, depth_target_size, interpolation=cv2.INTER_LINEAR) #双线性插值
            if left_depth_crop.max() == left_depth_crop.min():
                continue
            np.save(osp.join(out_dir, which_set, 'depth', '%.5d_left.npy' % sample_id), left_depth_crop)

            ## modify the intrinsic matrix for two hands
            # left_matrix = camera_intrinsic_matrix.copy()
            # left_matrix[0, 2] = left_matrix[0, 2] - left_bbox[0]
            # left_matrix[1, 2] = left_matrix[1, 2] - left_bbox[1]

            ## get the resize scale
            xscale, yscale = 1. * target_size[0]/left_bbox[2], 1. * target_size[1]/left_bbox[3]

            ## save the pickle
            pickle_item = {
                            'xyz_original':left_xyz,
                            'uv_original':left_kp_uv,
                            'visible':left_visible,
                            'K_original':camera_intrinsic_matrix,
                            'bbox':left_bbox,
                            'xy_scale':np.array([xscale, yscale])
                        }
            to_pickle['%.5d_left' % sample_id] = pickle_item
        if right_bbox is not None:
            ## crop, resize and save two hands
            right_crop = crop_pad_im_from_bounding_rect(image, right_bbox)
            right_crop = cv2.resize(right_crop, target_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(osp.join(out_dir, which_set, 'crop', '%.5d_right.png' % sample_id), right_crop)

            ## crop, resize and save depth map for two hands
            right_depth_crop = crop_pad_im_from_bounding_rect(depth.reshape(depth.shape+(1,)), right_bbox)
            right_depth_crop = cv2.resize(right_depth_crop, depth_target_size, interpolation=cv2.INTER_LINEAR)
            if right_depth_crop.max() == right_depth_crop.min():
                continue
            np.save(osp.join(out_dir, which_set, 'depth', '%.5d_right.npy' % sample_id), right_depth_crop)

            ## modify the intrinsic matrix for two hands
            # right_matrix = camera_intrinsic_matrix
            # right_matrix[0, 2] = right_matrix[0, 2] - right_bbox[0]
            # right_matrix[1, 2] = right_matrix[1, 2] - right_bbox[1]

            ## get the resize scale
            xscale, yscale = 1. * target_size[0]/right_bbox[2], 1. * target_size[1]/right_bbox[3]

            ## save the pickle
            pickle_item = {
                            'xyz_original':right_xyz,
                            'uv_original':right_kp_uv,
                            'visible':right_visible,
                            'K_original':camera_intrinsic_matrix,
                            'bbox':right_bbox,
                            'xy_scale':np.array([xscale, yscale])
                        }
            to_pickle['%.5d_right' % sample_id] = pickle_item
    
    pickle.dump(to_pickle, f)
    f.close()
    f_anno.close()
    print('process done')

    f = open(osp.join(out_dir, which_set, 'RHD_%s.pickle' % which_set), 'rb')
    db = pickle.load(f)
    print(len(sorted(db.keys())))
    f.close()
