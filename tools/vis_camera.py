'''
camera extrinsics visualization tools
modified from https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py
'''

from __future__ import print_function
from utils.print_fn import log
# Python 2/3 compatibility

import numpy as np
import cv2 as cv
import os

from numpy import linspace
import matplotlib

# matplotlib.use('TkAgg')

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv


def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1, 1] = 0
    M[1, 2] = 1
    M[2, 1] = -1
    M[2, 2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))


def create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, f_scale]
    X_img_plane[0:3, 1] = [width, height, f_scale]
    X_img_plane[0:3, 2] = [width, -height, f_scale]
    X_img_plane[0:3, 3] = [-width, -height, f_scale]
    X_img_plane[0:3, 4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, -height, f_scale]
    X_triangle[0:3, 1] = [0, -2*height, f_scale]
    X_triangle[0:3, 2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]


def create_board_model(extrinsics, board_width, board_height, square_size, draw_frame_axis=False):
    width = board_width*square_size
    height = board_height*square_size

    # draw calibration board
    X_board = np.ones((4, 5))
    #X_board_cam = np.ones((extrinsics.shape[0],4,5))
    X_board[0:3, 0] = [0, 0, 0]
    X_board[0:3, 1] = [width, 0, 0]
    X_board[0:3, 2] = [width, height, 0]
    X_board[0:3, 3] = [0, height, 0]
    X_board[0:3, 4] = [0, 0, 0]

    # draw board frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [height/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, height/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, height/2]

    if draw_frame_axis:
        return [X_board, X_frame1, X_frame2, X_frame3]
    else:
        return [X_board]


def draw_camera(ax, camera_matrix, cam_width, cam_height, scale_focal,
                extrinsics,
                patternCentric=True,
                annotation=True):
    from matplotlib import cm

    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf

    X_moving = create_camera_model(
        camera_matrix, cam_width, cam_height, scale_focal)

    cm_subsection = linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [cm.jet(x) for x in cm_subsection]

    for idx in range(extrinsics.shape[0]):
        # R, _ = cv.Rodrigues(extrinsics[idx,0:3])
        # cMo = np.eye(4,4)
        # cMo[0:3,0:3] = R
        # cMo[0:3,3] = extrinsics[idx,3:6]
        cMo = extrinsics[idx]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4, j] = transform_to_matplotlib_frame(
                    cMo, X_moving[i][0:4, j], patternCentric)
            ax.plot3D(X[0, :], X[1, :], X[2, :], color=colors[idx])
            min_values = np.minimum(min_values, X[0:3, :].min(1))
            max_values = np.maximum(max_values, X[0:3, :].max(1))
        # modified: add an annotation of number
        if annotation:
            X = transform_to_matplotlib_frame(
                cMo, X_moving[0][0:4, 0], patternCentric)
            ax.text(X[0], X[1], X[2], "{}".format(idx), color=colors[idx])

    return min_values, max_values

##### TO CREATE A SERIES OF PICTURES
 
def make_views(ax,angles,elevation=None, width=4, height = 3,
                prefix='tmprot_',**kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created. 
     
    Returns: the list of files created (for later removal)
    """
     
    files = []
    ax.figure.set_size_inches(width,height)
     
    for i,angle in enumerate(angles):
     
        ax.view_init(elev = elevation, azim=angle)
        fname = '%s%03d.jpeg'%(prefix,i)
        ax.figure.savefig(fname)
        files.append(fname)
     
    return files
 

##### TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION
 
def make_movie(files,output, fps=10,bitrate=1800,**kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """
     
    output_name, output_ext = os.path.splitext(output)
    command = { '.mp4' : 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                         %(",".join(files),fps,output_name,bitrate)}
                          
    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s'%(output_name,fps,output)
     
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])
 
 
 
def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """
     
    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              %(delay,loop," ".join(files),output))
 
 
 
 
def make_strip(files,output,**kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """
     
    os.system('montage -tile 1x -geometry +0+0 %s %s'%(" ".join(files),output))
     
     
     
##### MAIN FUNCTION
 
def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax
     
    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """
         
    output_ext = os.path.splitext(output)[1]
 
    files = make_views(ax,angles, **kwargs)
     
    D = { '.mp4' : make_movie,
          '.ogv' : make_movie,
          '.gif': make_gif ,
          '.jpeg': make_strip,
          '.png':make_strip}
           
    D[output_ext](files,output,**kwargs)
     
    for f in files:
        os.remove(f)
     
 
##### EXAMPLE
 
# if __name__ == '__main__':
 
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y, Z = axes3d.get_test_data(0.05)
#     s = ax.plot_surface(X, Y, Z, cmap=cm.jet)
#     plt.axis('off') # remove axes for visual appeal
     
#     angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
 
#     # create an animated gif (20ms between frames)
#     rotanimate(ax, angles,'movie.gif',delay=20) 
 
#     # create a movie with 10 frames per seconds and 'quality' 2000
#     rotanimate(ax, angles,'movie.mp4',fps=10,bitrate=2000)
 
#     # create an ogv movie
#     rotanimate(ax, angles, 'movie.ogv',fps=10)

def visualize(camera_matrix, extrinsics,label='', pts3d=None):

    ########################    plot params     ########################
    cam_width = 5*0.064/2     # Width/2 of the displayed camera.
    cam_height = 5*0.048/2    # Height/2 of the displayed camera.
    scale_focal = 4*40        # Value to scale the focal length.

    ########################    original code    ########################
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")
    ax.set_aspect("auto")

    min_values, max_values = draw_camera(ax, camera_matrix, cam_width, cam_height,
                                         scale_focal, extrinsics, True, False)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')

    if pts3d is not None:
        ax.scatter(pts3d[:,0], pts3d[:,1], -pts3d[:,2],marker='.', c='k')
    plt.savefig(f'viz/camera_extrinsics_{label}.png')
    angles=np.linspace(0,360,21)[:-1]
    rotanimate(ax,angles,f'viz/camera_extrinsics_{label}.gif',fps=10)
    plt.close()
    # plt.show()
    log.info('Done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_id", type=int, default=40)
    args = parser.parse_args()
    
    log.info(__doc__)
    # NOTE: jianfei: 20210722 newly checked. The coordinate is correct.
    #       note that the ticks on (-y) means the opposite of y coordinates.
    
    ########################    modified: example code    ########################
    from dataio.DTU import SceneDataset
    import torch
    train_dataset = SceneDataset(
        train_cameras=False,
        data_dir='./data/DTU/scan{}'.format(scan_id=args.scan_id))
    c2w = torch.stack(train_dataset.c2w_all).data.cpu().numpy()
    extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
    camera_matrix = next(iter(train_dataset))[1]['intrinsics'].data.cpu().numpy()


    # import pickle
    # data = pickle.load(open('./dev_test/london/london_siren_si20_cam.pt', 'rb'))
    # c2ws = data['c2w']
    # extrinsics = np.linalg.inv(c2ws)
    # camera_matrix = data['intr']
    visualize(camera_matrix, extrinsics)
    cv.destroyAllWindows()
