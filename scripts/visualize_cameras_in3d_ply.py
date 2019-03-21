import numpy as np
import glob
import os
import ntpath


def writePlyFile(file_name, vertices, colors):
    ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                   '''
    vertices = np.hstack([vertices, colors])
    with open(file_name, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

def generate_interpolated_3d_pts(pt1, pt2, num_pts=50):
    return pt1[np.newaxis, :] + ((pt2 - pt1) / num_pts)[np.newaxis,:] * np.linspace(0, num_pts, num_pts)[:, np.newaxis] 

def get_rot_mat(a_x, a_y, a_z):
    Rx = np.array([ [1           ,            0,            0],
                    [0           , np.cos(a_x) , -np.sin(a_x)],
                    [0           , np.sin(a_x) ,  np.cos(a_x)]], dtype=np.float)

    Ry = np.array([ [ np.cos(a_y),            0, np.sin(a_y) ],
                    [0           ,            1,            0],
                    [-np.sin(a_y),            0, np.cos(a_y) ]], dtype=np.float)

    Rz = np.array([ [ np.cos(a_z), -np.sin(a_z),            0],
                    [ np.sin(a_z),  np.cos(a_z),            0],
                    [0           ,            0,            1]], dtype=np.float)
    return np.matmul(Rz, np.matmul(Ry, Rx))

def gen_oriented_camera_3d_pts(camera_R, camera_origin, scale=1, color=[0, 255, 0]):
    yoff = 0.3 * scale
    zoff = 0.2 * scale
    x = 0.5 * scale
    R = get_rot_mat(np.pi/2, 0, 0)
    A = np.matmul(R, np.matmul(camera_R, np.array([x, yoff, -zoff]))) + camera_origin
    B = np.matmul(R, np.matmul(camera_R, np.array([x, -yoff,-zoff]))) + camera_origin
    C = np.matmul(R, np.matmul(camera_R, np.array([x, -yoff,zoff]))) + camera_origin
    D = np.matmul(R, np.matmul(camera_R, np.array([x, yoff, zoff]))) + camera_origin


    pointsAB = generate_interpolated_3d_pts(A, B)
    pointsBC = generate_interpolated_3d_pts(B, C)
    pointsCD = generate_interpolated_3d_pts(C, D)
    pointsDA = generate_interpolated_3d_pts(D, A)
    pts_view = np.concatenate([pointsAB, pointsBC, pointsCD, pointsDA], axis=0)
    #pts_view_color = np.ones(shape=[pts_view.shape[0], 3]) * np.array([[0, 0, 255]])
    pts_view_color = np.ones(shape=[pts_view.shape[0], 3]) * np.array([color])
    pointsCA = generate_interpolated_3d_pts(camera_origin, A)
    pointsCB = generate_interpolated_3d_pts(camera_origin, B)
    pointsCC = generate_interpolated_3d_pts(camera_origin, C)
    pointsCD = generate_interpolated_3d_pts(camera_origin, D)
    pts_body = np.concatenate([pointsCA, pointsCB, pointsCC, pointsCD], axis=0)
    pts_body_color = np.ones(shape=[pts_body.shape[0], 3]) * np.array([color])
    return np.concatenate([pts_view, pts_body], axis=0), np.concatenate([pts_view_color, pts_body_color], axis=0) 

def assert_rotation_matrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def quaternion_to_rotation_matrix(q_f):
    #q_f=quaternion.as_float_array(q)
    r=q_f[0]
    i=q_f[1]
    j=q_f[2]
    k=q_f[3]
    s = 1.0/(r*r+i*i+j*j+k*k)
    R = np.array([[1-2*s*(j*j+k*k), 2*s*(i*j-k*r),          2*s*(i*k+j*r)],
                  [2*s*(i*j+k*r),   1-2*s*(i*i+k*k),        2*s*(j*k-i*r)],
                  [2*s*(i*k-j*r),   2*s*(j*k+i*r),          1-2*s*(i*i+j*j)]],dtype=np.float64)

    assert_rotation_matrix(R)
    return R


if __name__ == "__main__":

    #npz_dir = "/data/cvfs/mttt2/RUNS/localsegPose/part34/newLoss_posenet2_2019_03_19_22.40/full_npz/Train_output/"
    #npz_dir_val = "/data/cvfs/mttt2/RUNS/localsegPose/part34/newLoss_posenet2_2019_03_19_22.40/full_npz/Val_output/"
    #npz_dir_gt = "/data/cvfs/ib255/shared_file_system/derivative_datasets/camvid_360_3d_data_final_sequences/camvid_360_final/part_0304_seq_016E5_P2_R94_08010_10380/posenet_info/" 

    # npz_dir = "/data/cvfs/mttt2/RUNS/localsegPose/part34/newLoss_posenet2_2019_03_19_22.40/oppSide/Train_output/"
    # npz_dir_val = "/data/cvfs/mttt2/RUNS/localsegPose/part34/newLoss_posenet2_2019_03_19_22.40/oppSide/Val_output/"

    npz_dir = "/data/cvfs/mttt2/RUNS/localsegPose/part34/newLoss_posenet2_2019_03_19_22.40/full_gt/Train_output/"
    npz_dir_val = "/data/cvfs/mttt2/RUNS/localsegPose/part34/newLoss_posenet2_2019_03_19_22.40/full_gt/Val_output/"

    out_dir = "./temp/camera3d_ply_test/"

    cams_3dpts = []
    cams_3dpts_colors = []
    file_list = sorted(glob.glob(npz_dir + "*.npz"))
    for i, filename in enumerate(file_list): 
        data = np.load(filename)
        r = quaternion_to_rotation_matrix(data["rotation"])
        origin = data["translation"].flatten()
        p, c = gen_oriented_camera_3d_pts(r, origin, scale=10, color=[255, 0, 0])
        cams_3dpts.append(p)
        cams_3dpts_colors.append(c)

        r = quaternion_to_rotation_matrix(data["rotation_gt"])
        origin = data["translation_gt"].flatten()
        p, c = gen_oriented_camera_3d_pts(r, origin, scale=10, color=[0, 255, 0])
        cams_3dpts.append(p)
        cams_3dpts_colors.append(c)


    #file_list = sorted(glob.glob(npz_dir_gt + "*.npz"))
    #for i, filename in enumerate(file_list): 
    #    data_gt = np.load(filename)
    #    R = quaternion_to_rotation_matrix(data_gt["Q_posenet"])
    #    origin = -R.T.dot(data_gt["T_opensfm"].flatten())
    #    p, c = gen_oriented_camera_3d_pts(R, origin, scale=11, color=[0, 0, 255])
    #    cams_3dpts.append(p)
    #    cams_3dpts_colors.append(c)
    
    file_list = sorted(glob.glob(npz_dir_val + "*.npz"))
    for i, filename in enumerate(file_list): 
        data = np.load(filename)
        r = quaternion_to_rotation_matrix(data["rotation"])
        origin = data["translation"].flatten()
        p, c = gen_oriented_camera_3d_pts(r, origin, scale=10, color=[0, 0, 255])
        cams_3dpts.append(p)
        cams_3dpts_colors.append(c)

        '''
        r = quaternion_to_rotation_matrix(data["rotation_gt"])
        origin = data["translation_gt"].flatten()
        p, c = gen_oriented_camera_3d_pts(r, origin, scale=10, color=[0, 255, 255])
        cams_3dpts.append(p)
        cams_3dpts_colors.append(c)
        '''


    cams_3dpts = np.concatenate(cams_3dpts, axis=0)
    cams_3dpts_colors = np.concatenate(cams_3dpts_colors, axis=0)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    name = npz_dir.split("/")[-3]

    writePlyFile(out_dir + "/{}_cameras3d.ply".format(name), cams_3dpts, cams_3dpts_colors)
