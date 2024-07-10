from transformations import *
from scipy.stats import expon


def add_noise_to_rows(array, noise_level):
    """
    Apply noise to each row of a NumPy array.

    Parameters:
    - array: NumPy array where each row represents a data point.
    - noise_level: Magnitude of noise to apply to each row.

    Returns:
    - noisy_array: NumPy array with noise added to each row.
    """
    # Determine the shape of the input array
    num_rows, num_cols = array.shape

    # Generate noise for each row
    noise = np.random.normal(scale=noise_level, size=(num_rows, num_cols))

    # Add noise to each row
    noisy_array = array + noise
    return noisy_array


def generate_waypath(params):
    """
    Generates a set of waypaths and corresponding way-points based on the parameters given.
    Parameters:
    - params: dictionary of the parameters for way-point generation

    Returns:
    - new_waypaths: List of returned waypaths between the way-points.
    - new_waypts:  List of returned way-points
    """
    # Sample number of way-points
    n_points = np.random.poisson(lam=params["waypts_lambda_offset"]) + params["min_waypts"]

    # Chance to be planar increases with number of way points
    prob_coplanar, coplanar_dim = expon.cdf(n_points - 2, scale=params["coplanar_scale"]), -1
    if np.random.uniform() < prob_coplanar: coplanar_dim = np.random.randint(0, 3)
    use_screw = False
    if n_points == 2 and np.random.uniform() > params["screw_prob"]: use_screw = True

    # Select way-points position change
    distances = np.random.uniform(0, params["max_obj_scale"], size=n_points)
    way_point_pos = random_unit_vectors_3d(n_points) * np.expand_dims(distances, -1)
    if coplanar_dim > -1: way_point_pos[:, coplanar_dim] = 0

    # Select way-points orientation change
    way_point_ori = []
    random_start = R.random().as_quat()
    for i in range(n_points):
        if coplanar_dim < 0 and np.random.uniform() > params["keep_ori_prob"]: random_start = R.random().as_quat()
        way_point_ori.append(random_start)

    # Generate waypaths
    new_waypts = np.concatenate([way_point_pos, way_point_ori], axis=-1)
    [lower_radius, upper_radius] = params["arc_radius_range"]
    wp_params = {'log_arc_radius_range': [np.log(lower_radius), np.log(upper_radius)],
                 'tilt_deg': params["arc_tilt_deg"],
                 'radius_noise': params["arc_radius_noise"], 'tilt_noise': params["arc_tilt_noise"]}
    new_waypaths = []
    for i in range(n_points - 1):
        # Generate between points
        pt1, pt2 = new_waypts[i], new_waypts[i + 1]
        dir_vector = pt2[:3] - pt1[:3]
        interval_steps = np.random.randint(params["min_steps"], params["max_steps"])
        if use_screw: interval_steps *= params["screw_step_mult"]
        wpath = ArcWayPath(wp_params, interval_steps)
        wpath.sample_params()
        sampled_pos = wpath.draw(pt2[:3], pt1[:3])
        sampled_ori = apply_screw(pt1[3:], dir_vector, interval_steps) if use_screw else quaternion_slerp(pt1[3:],
                                                                                                          pt2[3:],
                                                                                                          interval_steps)
        if coplanar_dim > -1: sampled_pos[:, coplanar_dim] = 0
        way_path_seg = np.concatenate([sampled_pos, sampled_ori], axis=-1)
        new_waypaths.append(way_path_seg)
    return new_waypaths, new_waypts


class ArcWayPath():
    '''
    Initializes, samples and generates arc-like waypaths.

    Attributes:
        steps (int): controls the timestep resolution of the arc sampled.
        arc_radius_mean (float): mean of the arc radius around which new waypaths are sampled.
        tilt_deg_mean (float): mean of the arc tilt degree around which new waypaths are sampled.
        arc_radius_noise (float): sampling noise for arc radius.
        tilt_noise (float): sampling noise for arc tilt degree.

    Methods:
        sample_params(): sample new parameters based on initial parameter and a degree of noise.
        draw(start_point, end_point): draw an arc connecting the two given way-points.
    '''

    def __init__(self, params, steps):
        self.params = params
        self.steps = steps
        self.arc_radius_mean = np.exp(np.random.uniform(*self.params["log_arc_radius_range"]))
        self.tilt_deg_mean = np.random.uniform(*self.params["tilt_deg"])
        self.arc_radius_noise = self.params["radius_noise"]
        self.tilt_noise = self.params["tilt_noise"]
        self.sample_params()

    def sample_params(self):
        """
        Sample new arc parameters based on initial parameter and a degree of noise.
        """
        # Trajectory property randomization
        self.arc_radius = np.random.normal(self.arc_radius_mean, self.arc_radius_noise)
        self.tilt_deg = np.random.normal(self.tilt_deg_mean, self.tilt_noise)

    def draw(self, start_point, end_point):
        """
        Draws a trajectory from start_point to end_point.
        Parameters:
        - start_point: array of 3D for start point
        - end_point: array of 3D for end point
        Return:
        - points_on_arc: array of 3D points on arc
        """
        x_axis, y_axis, z_axis = pairwise_constrained_axis3d(start_point, end_point, up_axis=2)
        rot = R.from_rotvec(x_axis * self.tilt_deg)
        new_y_axis, new_z_axis = rot.apply(y_axis), rot.apply(z_axis)
        direction_vector = end_point - start_point
        direction_norm = np.linalg.norm(direction_vector)
        # Generate points along the arc
        deg = max(0, np.arcsin(1 / (2 * self.arc_radius)))
        t = np.linspace(np.pi / 2 - deg, np.pi / 2 + deg, self.steps)

        points_on_arc = np.outer(self.arc_radius * np.cos(t), x_axis) + \
                        np.outer(self.arc_radius * np.sin(t), new_z_axis)
        points_on_arc *= direction_norm

        # Translate points to be centered at the midpoint of start and end points
        midpoint = (start_point + end_point) / 2
        offset = np.cos(deg) * self.arc_radius * direction_norm * new_z_axis
        points_on_arc += np.expand_dims(midpoint - offset, 0)
        return points_on_arc


class Object3D:
    def __init__(self, size, waypaths, waypoints):
        self.size = size
        self.waypaths = waypaths
        self.waypoints = waypoints
