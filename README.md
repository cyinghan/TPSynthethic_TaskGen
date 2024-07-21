# Task-Parameterized Synthetic Task Generator

Provides a generator class that automates the generation of synthetic tasks defined by a set of distribution parameters and used for multi-task imitation learning.

## Description
The TaskGenerator class can define a set of unique tasks, randomly initialize the object configurations, and produce the corresponding task trajectories (See sample_generation.ipynb for example task generation).
The task parameters for TaskGenerator is required to control the charactoristic of the output tasks and includes:
* <ins>waypoints_range</ins>: a range for uniformly sampling the number of way-points per object.
* <ins>step_range</ins>: step range use for uniformly sampling the number of steps per way-path.
* <ins>obj_num_range</ins>: range for uniformly sampling the number of object per task.
* <ins>obj_waypoint_bound</ins>: max distance the way-points can be from the object center.
* <ins>coplanar_scale</ins>: scale parameter controlling whether coplanar way-paths should be generated depending on the number of way-points.
* <ins>keep_ori_prob</ins>: controls if orientation between way-points is maintained.
* <ins>screw_prob</ins>: controls if screwing motion is performed for a way-path with 2 way-points.
* <ins>screw_step_mult</ins>: multipler for the sampled screwing motion steps as longer timestep is require to have a detailed resolution of the motion.
* <ins>arc_radius_range</ins>: mean of the arc radius around which new waypaths are sampled.
* <ins>arc_tilt_radian</ins>: mean of the arc tilt radian around which new waypaths are sampled.
* <ins>arc_radius_noise</ins>: sampling noise for arc radius.
* <ins>arc_tilt_noise</ins>: sampling noise for arc tilt radian.
* <ins>start_point</ins>: the provided starting pose for end-effector.

'''
from synthetic_task_gen.create_task import *

NUM_DEMOS = 10
UNIQUE_OBJS = ["obj0", "obj1", "obj2"]

# Plot individual randomized waypath
sample_params = {"waypoints_range":[2, 5], "obj_waypoint_bound": 150, "coplanar_scale": 2,
                 "keep_ori_prob": 0.6, "screw_prob": 0.4, "screw_step_mult": 2, "step_range": [15, 30],
                 "obj_num_range": [1, 2], "arc_radius_range": [0.8, 150], 'arc_tilt_radian': [-np.pi, np.pi],
                 "arc_radius_noise": 0.001, 'arc_tilt_noise': 0.001, 
                 "start_point": [-400, -400, 100, 0, 0, 0, 1]}
                 
# Generate tasks using the provided task parameters
generator = TaskGenerator(sample_params, UNIQUE_OBJS, n_tasks=tsize)

# Generate demonstration samples
generated_samples = generator.sample_demos(NUM_DEMOS)
'''
