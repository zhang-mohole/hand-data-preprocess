from __future__ import print_function, unicode_literals

import pickle
import os
import scipy.misc
import struct


def read_data(path_to_db):
    # load annotations of this set
    with open(os.path.join(path_to_db, set, 'anno_%s.pickle' % set), 'rb') as fi:
        anno_all = pickle.load(fi)

        num_samples = len(anno_all.items())
        for sample_id, anno in anno_all.items():
            # load data
            image = scipy.misc.imread(os.path.join(path_to_db, set, 'color', '%.5d.png' % sample_id))
            mask = scipy.misc.imread(os.path.join(path_to_db, set, 'mask', '%.5d.png' % sample_id))

            # get info from annotation dictionary
            kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
            kp_visible = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
            kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
            camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

            # do what you want

if __name__ == "__main__":
    pass
