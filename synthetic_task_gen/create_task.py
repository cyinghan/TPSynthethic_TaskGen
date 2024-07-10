import torch
import matplotlib.pyplot as plt
from create_waypath import *
from scipy.interpolate import CubicSpline


def random_table_arrangment(objects, dims=(600, 800, 0)):
    """
    Randomize the placement of objects while avoiding overlaps.
    Parameters:
    - objects: List of objects to be considered.
    - dims:  dimensions of the task space.

    Returns:
    - randomized objects positions.
    """
    # Define the dimensions of the table
    width = dims[0] / 2
    length = dims[1] / 2
    height = dims[2] / 2
    positions = []
    for obj in objects:
        # Randomly choose a position within the table dimensions
        x = random.uniform(-width, width)
        y = random.uniform(-length, length)
        z = random.uniform(-height, height) + obj.size

        # Check if the object overlaps with any other object
        while any(intersects((x, y, z), pos, obj.size) for pos in positions):
            x = random.uniform(-width, width)
            y = random.uniform(-length, length)

        # Add the position to the list of positions
        positions.append([x, y, z])
    return np.array(positions)


def intersects(pos1, pos2, dist):
    """
    Check if the two given positions intersect.
    """
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    if (x1 < x2 + dist and x1 + dist > x2 and
            y1 < y2 + dist and y1 + dist > y2 and
            z1 < z2 + dist and z1 + dist > z2):
        return True
    else:
        return False


def generate_pairing_target(n):
    """
    Generates n random pairing target with -1/1 for previous/next target and 0 for none.
    Parameters:
    - n (int): number of objects
    """
    output = [round(np.random.uniform(-1, 1)) for _ in range(n)]
    if output[0] < 0: output[0] = 0
    if output[-1] > 0: output[-1] = 0
    return output


def create_tags(objs):
    """
    Create a dictionary of one-hot vectors for a list of unique objects.
    """
    one_hots = torch.eye(len(objs))
    tag_dict = {}
    for i, obj in enumerate(objs):
        tag_dict[obj] = one_hots[i]
    return tag_dict


def generate_demo(task_info):
    """
    Generates a random demonstration based on the task information.

    Parameters:
    - task_info: a dictionary of the task information containing the object and waypath information.
    - start_point: start point in the new demonstration

    Returns:
    - local_waypaths: a list of trajectories and local waypaths in sequential order.
    - local_waypoints: a list of the local way-points in sequential order.
    - rotations: a list of the random rotation for the objects in the demonstration.
    - obj_centers: a list of the randomly initialized object centers in the demonstration.
    """
    n_objs = len(task_info["object_list"])
    # get randomized object locations
    obj_centers = random_table_arrangment(task_info["object_list"])
    ref_frame_target = task_info["pairwise_frame_target"]

    # collect local object waypaths
    local_waypaths, local_waypoints = [], [[task_info["start_point"]]]
    rotations = []
    for i in range(n_objs):
        obj = task_info["object_list"][i]
        obj_offset = obj_centers[i][None, :]
        # get randomize individual object z-axis rotations
        if ref_frame_target[i] == 1:
            axis_rot_3d = pairwise_constrained_axis3d(obj_centers[i], obj_centers[i + 1], up_axis=2)
            sampled_rot = R.from_matrix(axis_rot_3d)
        elif ref_frame_target[i] == -1:
            axis_rot_3d = pairwise_constrained_axis3d(obj_centers[i], obj_centers[i - 1], up_axis=2)
            sampled_rot = R.from_matrix(axis_rot_3d)
        else:
            sampled_rot = single_dim_rotation(dim=2)
        new_waypath_out = [apply_rotation(traj, sampled_rot) for traj in obj.waypaths]
        new_waypt_out = apply_rotation(obj.waypoints, sampled_rot)
        rotations.append(sampled_rot.as_quat())

        # saved randomized object configuration.
        new_waypt_out[:, :3] += obj_offset
        for j in range(len(new_waypath_out)):
            new_waypath_out[j][:, :3] += obj_offset
        local_waypaths = local_waypaths + new_waypath_out
        local_waypoints.append(new_waypt_out)

        # interpolate between the start points and the end points between local waypath objects
        interval_steps = task_info["transit_trajs"][i]
        obj_connect_end_pts = []
        obj_connect_end_pts.extend(local_waypoints[-2])
        nth_chunk = len(obj_connect_end_pts) - 1
        obj_connect_end_pts.extend(local_waypoints[-1])
        obj_connect_end_pts = np.array(obj_connect_end_pts)

        inter_traj_pos = []
        n_pts = obj_connect_end_pts.shape[0]
        timesteps = list(range(0, n_pts))
        for j in range(3):
            cs = CubicSpline(timesteps, obj_connect_end_pts[:, j])
            interpolated_timesteps = np.linspace(nth_chunk, nth_chunk + 1, interval_steps)
            inter_traj_pos.append(cs(interpolated_timesteps))
        inter_traj_ori = quaternion_slerp(local_waypoints[-2][-1][3:], local_waypoints[-1][0][3:], interval_steps)
        inter_traj_pos = np.array(inter_traj_pos).T
        local_waypaths.insert(-len(new_waypt_out) + 1, np.concatenate([inter_traj_pos, inter_traj_ori], axis=-1))
    return local_waypaths, local_waypoints, np.array(rotations), np.array(obj_centers)


def generate_task(params, paired_ref_prob_func=generate_pairing_target):
    """
    Generates a task based on the distribution parameters given.

    Parameters:
    - params: dictionary of the distribution parameters for task generation
    - paired_ref_prob_func: a function that randomly determines if the reference frame is paired wise or not

    Returns:
    - task_info: a dictionary of the randomly generated task parameters.
    """
    task_info = {"start_point": params["start_point"], "object_list": [], "transit_trajs": [],
                 "num_objs": random.randint(*params["obj_num_range"])}
    task_info["pairwise_frame_target"] = paired_ref_prob_func(task_info["num_objs"])
    task_info.update(params)
    for i in range(task_info["num_objs"]):
        waypath_out, waypt_out = generate_waypath(task_info)
        obj = Object3D(task_info["obj_waypoint_bound"], waypath_out, waypt_out)
        transit_len = np.random.randint(*task_info["step_range"])
        task_info["object_list"].append(obj)
        task_info["transit_trajs"].append(transit_len)
    return task_info


class TaskGenerator(object):
    '''
    Task generation class.

    Attributes:
        task_info_sets (dict): dictionary containing tasks information.
        objects (list): possible unique objects.
        obj_tags: one-hot object tags.
        task_tags: one-hot task tags.

    Methods:
        sample_demos(num_demos): sample demonstrations for all tasks.
    '''

    def __init__(self, task_params, object_types, n_tasks):
        self.task_info_sets = {}
        for i in range(n_tasks):
            self.task_info_sets[f"task{i}"] = generate_task(task_params)
        self.task_names = list(self.task_info_sets.keys())
        self.objects = object_types
        self.n_object = len(self.objects)
        self.obj_tags = create_tags(self.objects)
        self.task_tags = create_tags(self.task_names)

    def sample_demos(self, num_demos):
        """
        Generate task dataset based on provided parameters.
        """
        # Load all task data and create data test/training splits
        task_datasets = {}
        for tname, task_info in self.task_info_sets.items():
            all_obj_seqs, all_traj_seqs = [], []

            for j in range(num_demos):
                local_wpaths, local_wpts, local_rots, obj_centers = generate_demo(task_info)
                traj = np.concatenate(local_wpaths)
                traj_len = traj.shape[0]
                obj_poses = np.concatenate([obj_centers, local_rots], axis=-1)
                obj_poses = np.vstack([task_info["start_point"], obj_poses])
                obj_buffer = np.zeros([traj_len, self.n_object])
                task_buffer = self.task_tags[tname].repeat([traj_len, 1])
                smooth_quaternions(traj[:, 3:])
                new_traj_data = np.concatenate([traj, obj_buffer, task_buffer], axis=1)
                obj_count = obj_poses.shape[0]
                obj_tag_addon = np.array([self.obj_tags[self.objects[i]] for i in range(obj_count)])
                obj_task_addon = self.task_tags[tname].repeat([obj_count, 1])
                new_obj_seq = np.concatenate([obj_poses, obj_tag_addon, obj_task_addon], axis=-1)
                all_obj_seqs.append(new_obj_seq)
                all_traj_seqs.append(new_traj_data)

            task_datasets[tname] = [all_obj_seqs, all_traj_seqs]
        return task_datasets


def display_trajectory(trajectories, obj_position):
    """
    Display trajectory(blue) and object positions(red) as a matplotlib figure.
    Parameters:
    - trajectory: a NumPy array representing the trajectory.
    - obj_position: a NumPy array representing the object positions.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    line_config = {"linestyle": "--", "color": 'blue', "alpha": 0.5}
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], **line_config)
    for i, pos in enumerate(obj_position):
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], marker="x", color="red", label="start point")
        ax.scatter(pos[1:, 0], pos[1:, 1], pos[1:, 2], marker="o", color="red", label="object center")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    pts = np.concatenate(trajectories)
    center, bound = pts.mean(axis=0), (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    ax.set_xlim([center[0] - bound, center[0] + bound])
    ax.set_ylim([center[1] - bound, center[1] + bound])
    ax.set_zlim([center[2] - bound, center[2] + bound])
    plt.show()


def draw_3d_axes(ax, origin, rot_mat, length=1.0):
    """
    Draw the 3D axis at origin and given a rotation matrix.
    Parameters:
    - ax: 3D axes object.
    - origin: Origin of the 3D axis.
    - rot_mat: 3x3 rotation matrix.
    - length (float): Length of the 3D axis.
    """
    # Origin of the axes
    ox, oy, oz = origin
    # X axis
    ax.quiver(ox, oy, oz, length * rot_mat[0][0], length * rot_mat[0][1], length * rot_mat[0][2], color='r',
              label='X axis')
    # Y axis
    ax.quiver(ox, oy, oz, length * rot_mat[1][0], length * rot_mat[1][1], length * rot_mat[1][2], color='g',
              label='Y axis')
    # Z axis
    ax.quiver(ox, oy, oz, length * rot_mat[2][0], length * rot_mat[2][1], length * rot_mat[2][2], color='b',
              label='Z axis')


class Object3D:
    def __init__(self, size, waypaths, waypoints):
        self.size = size
        self.waypaths = waypaths
        self.waypoints = waypoints
