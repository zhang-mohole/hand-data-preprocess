import pickle
import cv2
import os
import numpy as np

def draw_joints(uv, kp_visible, img, out_path):
    '''
    to draw the joints and bbox of one frame
        uv: 21 joints 2d coordinates (21, 2)
        img: initial image read by cv2 [480, 640]
    '''
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    # draw joints
    for i, point in enumerate(uv):
        if kp_visible[i]:
            cv2.circle(img, (int(point[0]),int(point[1])), point_size, point_color, thickness)
    cv2.imwrite(out_path, img)
    
def visulize_data(root_path):
    anno = pickle.load(open(os.path.join(root_path, 'RHD_evaluation.pickle'), 'rb'))
    anno_key = sorted(anno.keys())

    for i in range(100):
        name = anno_key[i]
        label = anno[name]

        image_path   = os.path.join(root_path, 'crop', name + '.png')
        img = cv2.imread(image_path)

        depth_path = os.path.join(root_path, 'depth', name + '.npy')
        depthmap = np.load(depth_path)

        original_coor2d = label['uv_original']
        coor3d = label['xyz_original']
        original_K = label['K_original']
        bbox = label['bbox']
        xy_scale = label['xy_scale']
        kp_visible = label['visible']

        ## transform the coord and matrix
        coor2d = original_coor2d - np.tile(np.expand_dims(bbox[:2], axis=0), (21,1))
        coor2d[:, 0] *= xy_scale[0]
        coor2d[:, 1] *= xy_scale[1]
        matrix = original_K.copy()
        matrix[0, 2] -= bbox[0]
        matrix[1, 2] -= bbox[1]
        scale =[[xy_scale[0],    0,  0],
                [0,    xy_scale[1],  0],
                [0,              0,  1]]
        matrix = np.matmul(scale, matrix)
        # matrix = np.linalg.inv(matrix) #take the inversion of matrix

        out_path = os.path.join(root_path, 'vis_out', name+'.png')
        draw_joints(coor2d, kp_visible, img, out_path)
        print(name, 'processed')

if __name__ == "__main__":
    visulize_data('../RHD_data/processed/evaluation')