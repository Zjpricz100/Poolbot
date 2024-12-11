import numpy as np

def compute_transformation(R_A, t_A, R_B, t_B):
    # Construct homogeneous transformation matrices
    T_A = np.eye(4)
    T_A[:3, :3] = R_A
    T_A[:3, 3] = t_A

    T_B = np.eye(4)
    T_B[:3, :3] = R_B
    T_B[:3, 3] = t_B

    # Compute the transformation from frame A to frame B
    T_A_inv = np.linalg.inv(T_A)
    T_A_to_B = np.dot(T_B, T_A_inv)

    return T_A_to_B

# Example: Replace these with actual detected poses
R_A = np.eye(3)  # Rotation matrix in frame A
t_A = np.array([0, 0, 0])  # Translation vector in frame A

R_B = np.eye(3)  # Rotation matrix in frame B
t_B = np.array([1, 1, 1])  # Translation vector in frame B

T_A_to_B = compute_transformation(R_A, t_A, R_B, t_B)
