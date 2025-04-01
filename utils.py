import numpy as np
from scipy import linalg
import os

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P


def DLTfrom3(P1, P2,P3, point1, point2,point3):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:],
         point3[1] * P3[2, :] - P3[1, :],
         P3[0, :] - point3[0] * P3[2, :],
        ]
    A = np.array(A).reshape((6,4))
    #print('A: ')
    #print(A)

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
    # U2, s2, Vh2 = linalg.svd(A, full_matrices=False)
    print('Triangulated point: ')
    print(Vh[-1,0:3]/Vh[-1,3])
    #print(Vh2[-1, 0:3] / Vh2[-1, 3])
    return Vh[-1,0:3]/Vh[-1,3]


def DLTfrom5(P1, P2,P3, P4, P5, point1, point2,point3, point4, point5):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:],
         point3[1] * P3[2, :] - P3[1, :],
         P3[0, :] - point3[0] * P3[2, :],
         point4[1] * P4[2, :] - P4[1, :],
         P4[0, :] - point4[0] * P4[2, :],
         point5[1] * P5[2, :] - P5[1, :],
         P5[0, :] - point5[0] * P5[2, :],
        ]
    A = np.array(A).reshape((10,4))
    #print('A: ')
    #print(A)

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
    # U2, s2, Vh2 = linalg.svd(A, full_matrices=False)
    print('Triangulated point: ')
    print(Vh[-1,0:3]/Vh[-1,3])
    #print(Vh2[-1, 0:3] / Vh2[-1, 3])
    return Vh[-1,0:3]/Vh[-1,3]


def DLT(points_2d, projections):
    """
    Perform Direct Linear Transform (DLT) to triangulate a 3D point from multiple 2D correspondences.

    Parameters:
    points_2d (list of np.array): List of 2D points from different views.
    projections (list of np.array): List of projection matrices for each view.

    Returns:
    np.array: Triangulated 3D point.
    """
    A = []
    for point, P in zip(points_2d, projections):
        A.append(point[1] * P[2, :] - P[1, :])
        A.append(P[0, :] - point[0] * P[2, :])
    A = np.array(A)
    _, _, Vh = linalg.svd(A, full_matrices=False)

    return Vh[-1,0:3]/Vh[-1,3]



def ransac_dlt(points_2d, projections, threshold=5.0, max_iterations=20):
    """
    Perform RANSAC to robustly estimate the 3D point using DLT.

    Parameters:
    points_2d (list of np.array): List of 2D points from different views.
    projections (list of np.array): List of projection matrices for each view.
    threshold (float): Distance threshold to determine inliers.
    max_iterations (int): Maximum number of iterations.

    Returns:
    np.array: Estimated 3D point.
    """
    best_inliers = []
    best_point_3d = None

    num_views = len(points_2d)
    all_indices = np.arange(num_views)

    for _ in range(max_iterations):
        # Randomly sample 2 points
        sample_indices = np.random.choice(all_indices, 3, replace=False)
        sampled_points = [points_2d[i] for i in sample_indices]
        sampled_projections = [projections[i] for i in sample_indices]

        # Estimate 3D point using DLT
        candidate_point_3d = DLT(sampled_points, sampled_projections)

        # Determine inliers
        inliers = []
        for i in range(num_views):
            reproj_point = projections[i] @ np.append(candidate_point_3d, 1)
            reproj_point /= reproj_point[2]
            reproj_point = reproj_point[:2]
            error = np.linalg.norm(reproj_point - points_2d[i])
            if error < threshold:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_point_3d = candidate_point_3d

    # Refine estimate using all inliers
    if best_inliers:
        inlier_points = [points_2d[i] for i in best_inliers]
        inlier_projections = [projections[i] for i in best_inliers]
        best_point_3d = DLT(inlier_points, inlier_projections)

    #print(best_point_3d)
    return best_point_3d


def calculate_3d_positions(frame_keypoints, projection_matrices):
    frame_p3ds = []
    for uv_points in zip(*frame_keypoints):
        p3d = DLT(uv_points, projection_matrices)
        frame_p3ds.append(p3d)
    #print(frame_p3ds)
    return frame_p3ds

def read_camera_parameters(camera_id):
  filepath = os.path.join('./cameraCal/camera_parameters/', 'camera' + str(camera_id) + '_intrinsics.dat')
  inf = open(filepath, 'r')

  cmtx = []
  dist = []

  line = inf.readline()
  for _ in range(3):
    line = inf.readline().split()
    line = [float(en) for en in line]
    cmtx.append(line)

  line = inf.readline()
  line = inf.readline().split()
  line = [float(en) for en in line]
  dist.append(line)

  return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder='./cameraCal/camera_parameters/'):
  filepath = os.path.join(savefolder, 'camera' + str(camera_id) + '_rot_trans.dat')
  inf = open(filepath, 'r')

  inf.readline()
  rot = []
  trans = []
  for _ in range(3):
    line = inf.readline().split()
    line = [float(en) for en in line]
    rot.append(line)

  inf.readline()
  for _ in range(3):
    line = inf.readline().split()
    line = [float(en) for en in line]
    trans.append(line)

  inf.close()
  return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
  filepath = os.path.join("./outputs/", str(filename))
  fout = open(filepath, 'w+')

  for frame_kpts in kpts:
    for kpt in frame_kpts:
      if len(kpt) == 2:
        fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
      else:
        fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

    fout.write('\n')
  fout.close()

#general rotation matrices
def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R


#calculate rotation matrix to take A vector to B vector
def Get_R(A,B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    #get products
    dotprod = np.sum(uA * uB)
    crossprod = np.sqrt(np.sum(np.square(np.cross(uA,uB)))) #magnitude

    #get new unit vectors
    u = uA
    v = uB - dotprod*uA
    v = v/np.sqrt(np.sum(np.square(v)))
    w = np.cross(uA, uB)
    w = w/np.sqrt(np.sum(np.square(w)))

    #get change of basis matrix
    C = np.array([u, v, w])

    #get rotation matrix in new basis
    R_uvw = np.array([[dotprod, -crossprod, 0],
                      [crossprod, dotprod, 0],
                      [0, 0, 1]])

    #full rotation matrix
    R = C.T @ R_uvw @ C
    #print(R)
    return R

#Same calculation as above using a different formalism
def Get_R2(A, B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    v = np.cross(uA, uB)
    s = np.sqrt(np.sum(np.square(v)))
    c = np.sum(uA * uB)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx@vx*((1-c)/s**2)

    return R


#decomposes given R matrix into rotation along each axis. In this case Rz @ Ry @ Rx
def Decompose_R_ZYX(R):

    #decomposes as RzRyRx. Note the order: ZYX <- rotation by x first
    thetaz = np.arctan2(R[1,0], R[0,0])
    thetay = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    thetax = np.arctan2(R[2,1], R[2,2])

    return thetaz, thetay, thetax

def Decompose_R_ZXY(R):

    #decomposes as RzRXRy. Note the order: ZXY <- rotation by y first
    thetaz = np.arctan2(-R[0,1], R[1,1])
    thetay = np.arctan2(-R[2,0], R[2,2])
    thetax = np.arctan2(R[2,1], np.sqrt(R[2,0]**2 + R[2,2]**2))

    return thetaz, thetay, thetax

if __name__ == '__main__':

    P2 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
