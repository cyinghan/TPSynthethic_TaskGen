import random
from transformations import *
from scipy.stats import expon


def generate_waypath(params):
    """
    Generates a set of waypaths and corresponding way-points based on the parameters given.
    Parameters:
    - params: dictionary of the parameters for way-point generation

    Returns:
    - new_waypaths: List of returned waypaths between the way-points.
    - new_waypoints:  List of returned way-points
    """
    # Sample number of way-points
    min_pt, max_pt = params["waypoints_range"]
    n_points = random.randint(min_pt, max_pt)

    # Chance to be planar increases with number of way points
    prob_coplanar, coplanar_dim = expon.cdf(n_points - 2, scale=params["coplanar_scale"]), -1
    if np.random.uniform() < prob_coplanar: coplanar_dim = random.randint(0, 2)
    use_screw = False
    if n_points == 2 and np.random.uniform() > params["screw_prob"]: use_screw = True

    # Select way-points position change
    distances = np.random.uniform(0, params["obj_waypoint_bound"], size=n_points)
    way_point_pos = random_unit_vectors_3d(n_points) * np.expand_dims(distances, -1)
    if coplanar_dim > -1: way_point_pos[:, coplanar_dim] = 0

    # Select way-points orientation change
    way_point_ori = []
    random_start = R.random().as_quat()
    for i in range(n_points):
        if coplanar_dim < 0 and np.random.uniform() > params["keep_ori_prob"]: random_start = R.random().as_quat()
        way_point_ori.append(random_start)

    # Generate waypaths
    new_waypoints = np.concatenate([way_point_pos, way_point_ori], axis=-1)
    [lower_radius, upper_radius] = params["arc_radius_range"]
    new_waypaths = []
    for i in range(n_points - 1):
        # Generate between points
        pt1, pt2 = new_waypoints[i], new_waypoints[i + 1]
        dir_vector = pt2[:3] - pt1[:3]
        interval_steps = np.random.randint(*params["step_range"])
        if use_screw: interval_steps *= params["screw_step_mult"]
        wpath = ArcWayPath(log_radius_range=[np.log(lower_radius), np.log(upper_radius)], tilt_radian_range=params["arc_tilt_radian"],
                           radius_noise=params["arc_radius_noise"], tilt_noise=params["arc_tilt_noise"],
                           steps=interval_steps)
        wpath.sample_params()
        sampled_pos = wpath.draw(pt2[:3], pt1[:3])
        sampled_ori = apply_screw(pt1[3:], dir_vector, interval_steps) if use_screw else quaternion_slerp(pt1[3:],
                                                                                                          pt2[3:],
                                                                                                          interval_steps)
        if coplanar_dim > -1: sampled_pos[:, coplanar_dim] = 0
        way_path_seg = np.concatenate([sampled_pos, sampled_ori], axis=-1)
        new_waypaths.append(way_path_seg)
    return new_waypaths, new_waypoints


class ArcWayPath():
    '''
    Defines and samples arc way-path properties.

    Attributes:
        steps (int): controls the timestep resolution of the arc sampled.
        arc_radius_mean (float): mean of the arc radius around which new waypaths are sampled.
        tilt_radian_mean (float): mean of the arc tilt radian around which new waypaths are sampled.
        arc_radius_noise (float): sampling noise for arc radius.
        tilt_noise (float): sampling noise for arc tilt radian.

    Methods:
        sample_params(): sample new parameters based on initial parameter and noise.
        draw(start_point, end_point): draw an arc connecting the two given way-points.
    '''

    def __init__(self, log_radius_range, tilt_radian_range, radius_noise, tilt_noise, steps):
        '''
        Initializes arc properties.

        Params:
        - log_radius_range ([float, float]): a range of logarithmic arc radius to be sampled from.
        - tilt_radian_range (float): a range of possible tilt angles to be sampled from.
        - radius_noise (float): sampling noise for arc radius.
        - tilt_noise (float): sampling noise for arc tilt.
        - steps (int): the arc timestep resolution.

        Methods:
            sample_params(): sample new parameters based on initial parameter and noise parameter.
            draw(start_point, end_point): draw an arc connecting the two given way-points.
        '''
        self.steps = steps
        self.arc_radius_mean = np.exp(np.random.uniform(*log_radius_range))
        self.tilt_radian_mean = np.random.uniform(*tilt_radian_range)
        self.arc_radius_noise = radius_noise
        self.tilt_noise = tilt_noise
        self.sample_params()

    def sample_params(self):
        # Trajectory property randomization
        self.arc_radius = max(0.5, np.random.normal(self.arc_radius_mean, self.arc_radius_noise))
        self.tilt_rad = np.random.normal(self.tilt_radian_mean, self.tilt_noise)

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
        rot = R.from_rotvec(x_axis * self.tilt_rad)
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



