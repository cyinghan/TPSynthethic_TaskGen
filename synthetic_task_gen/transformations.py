import numpy as np
from scipy.spatial.transform import Rotation as R


def vector_to_unit_vector(vector):
    magnitude = np.linalg.norm(vector)
    unit_vector = vector / magnitude
    return unit_vector


def random_unit_vectors_3d(n):
    """
    Generate random 3D unit vectors.

    Parameters:
    - n: Number of random unit vectors to generate.

    Returns:
    - unit_vectors: Array of generated unit vectors with shape (n, 3).
    """
    # Generate random numbers from a standard normal distribution
    random_numbers = np.random.normal(size=(n, 3))

    # Calculate the length of each vector
    lengths = np.linalg.norm(random_numbers, axis=1)

    # Normalize the vectors to obtain unit vectors
    unit_vectors = random_numbers / lengths[:, np.newaxis]
    return unit_vectors


def single_dim_rotation(dim=0):
    rot = R.random().as_quat()
    for i in range(3):
        if dim != i: rot[i] = 0
    rot = R.from_quat(rot)
    return rot


def apply_rotation(pts, rot):
    """
    Perform rotations on the given set of points.

    Parameters:
    - pts: NumPy array of shape (n, 3) containing generated points.
    - rot: scipy rotation applied to each points

    Returns:
    - new_pts: array of shape (n, 3) containing rotated points
    """
    new_pts = []
    for pt in pts:
        pos, ori = pt[:3], R.from_quat(pt[3:])
        new_pts.append(np.concatenate([rot.apply(pos), (rot * ori).as_quat()]))
    new_pts = np.array(new_pts)
    return new_pts


def apply_screw(start_ori, direction, steps, rot_rad=np.pi / 4):
    """
    Apply a constant rotation iteratively to the starting orientation for multiple steps.
    Parameters:
    - start_ori: quaternion representation of the starting orientations.
    - direction: vector representation of the rotation direction.
    - steps: number of steps to apply the rotation to.
    - rot_rad: rotation angle in radians per step.

    Returns:
    - rotated_ori: array of shape (n, 4) containing rotated points
    """
    rotation_vectors = np.array([rot_rad * vector_to_unit_vector(direction) for _ in range(steps)])
    rotated_ori = apply_rotation_waypath(start_ori, rotation_vectors)
    return rotated_ori


def apply_rotation_waypath(start_pose, rotation_vectors):
    """
    Apply a set of rotation iteratively to the starting orientation in sequence.
    Parameters:
    - start_pose: a quaternion representation of the starting pose.
    - rotation_vectors: array of shape (n, 3) containing rotation vectors.
    Return:
    - orientations: array of shape (n, 4) containing rotated points
    """
    orientations = []
    cur_orientation = R.from_quat(start_pose)
    for rotvec in rotation_vectors:
        # Calculate the rotation quaternion for this step
        rotation_quaternion = R.from_rotvec(rotvec)
        # Apply the rotation quaternion to the previous orientation
        cur_orientation = cur_orientation * rotation_quaternion
        orientations.append(cur_orientation.as_quat())
    return np.array(orientations)


def trajectory_transform(x, pos, quat):
    """
    Transform the trajectory given position and rotation
    Parameters:
    - x: Array of shape (n, 7) containing trajectory points.
    - pos: Array of shape (3) of the translation.
    - quat: Array of shape (4) of the rotation.
    """
    new_x = x.copy()
    rot = R.from_quat(quat)
    new_x[:, :3] = new_x[:, :3] - pos
    new_x[:, 3:7] = (rot.inv() * R.from_quat(new_x[:, 3:7])).as_quat()
    return new_x


def smooth_quaternions(quat_seq, threshold=-0.25):
    """
    Smooths the coefficient values of a list of quaternions using a threshold
     and the property that q=-q.
    Parameters:
    - quats (list): list of quaternions
    - threshold (float): threshold for smoothing
    """
    length = len(quat_seq)
    for i in range(1, length):
        if np.dot(quat_seq[i], quat_seq[i - 1]) < threshold:
            quat_seq[i] = -quat_seq[i]


def quaternion_slerp(q1, q2, n):
    """
    Perform spherical linear interpolation (slerp) between two quaternions.

    Parameters:
    - q1: NumPy array representing the first quaternion.
    - q2: NumPy array representing the second quaternion.
    - n: Number of time intervals for interpolation.

    Returns:
    - slerp_interp: List of quaternions representing the interpolated orientations.
    """
    # Ensure quaternions are normalized
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    # Compute angle between quaternions
    dot_product = np.dot(q1, q2)
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    if dot_product > 0.9995:
        # Linear interpolation if quaternions are close
        t = np.linspace(0, 1, n)[:, np.newaxis]
        slerp_interp = q1[np.newaxis, :] * (1 - t) + q2[np.newaxis, :] * t
    else:
        theta_0 = np.arccos(dot_product)
        sin_theta_0 = np.sin(theta_0)
        t = np.linspace(0, 1, n)
        sin_theta = np.sin(theta_0 * (1 - t)) / sin_theta_0
        sin_theta_prime = np.sin(theta_0 * t) / sin_theta_0
        slerp_interp = (
                q1[np.newaxis, :] * sin_theta[:, np.newaxis] + q2[np.newaxis, :] * sin_theta_prime[:, np.newaxis])
    return slerp_interp


def pairwise_constrained_axis3d(pos1, pos2, up_axis=0):
    '''
    Generates local object's 3D axis given pos1 and pos2 with x-axis parallel to the line between pos1 and pos2
    and z-axis perpendicular such that its unit vector's up-axis value is maximized.

    Parameters:
    ----------
    pos1: numpy.array
        A 3D numpy array representing position of object 1.
    pos2: numpy.array
        A 3D numpy array representing position of object 2.
    up_axis: int
        dimension of the z-axis' unit vector that should be maximized

    Returns:
    -------
        Three 3D unit vectors representing the direction xyz-axis are pointing.
    '''
    vec = pos2 - pos1
    x_axis_norm = vec / np.linalg.norm(vec)
    up = np.zeros(3)
    up[up_axis] = 1
    x_comp_of_up = np.dot(up, x_axis_norm) * x_axis_norm

    z_axis_norm = (up - x_comp_of_up) / np.linalg.norm(up - x_comp_of_up)
    y_axis = np.cross(z_axis_norm, x_axis_norm)
    y_axis_norm = y_axis / np.linalg.norm(y_axis)
    return (x_axis_norm, y_axis_norm, z_axis_norm)
