# Task-Parameterized Synthetic Task Generator

Provides a generator class that automates the generation of synthetic tasks defined by a set of distribution parameters and used for multi-task imitation learning.

## Description
The TaskGenerator class can define a set of unique tasks, randomly initialize the object configurations, and produce the corresponding task trajectories (See sample_generation.ipynb for example task generation).
The task parameters for TaskGenerator is required to control the charactoristic of the output tasks and includes:
* waypoints_range: a range for uniformly sampling the number of way-points per object.
* step_range: step range use for uniformly sampling the number of steps per way-path.
* obj_num_range: range for uniformly sampling the number of object per task.
* obj_waypoint_bound: max distance the way-points can be from the object center.
* coplanar_scale: scale parameter controlling whether coplanar way-paths should be generated depending on the number of way-points.
* keep_ori_prob: controls if orientation between way-points is maintained.
* screw_prob: controls if screwing motion is performed for a way-path with 2 way-points.
* screw_step_mult: multipler for the sampled screwing motion steps as longer timestep is require to have a detailed resolution of the motion.
* arc_radius_range: mean of the arc radius around which new waypaths are sampled.
* arc_tilt_radian: mean of the arc tilt radian around which new waypaths are sampled.
* arc_radius_noise: sampling noise for arc radius.
* arc_tilt_noise: sampling noise for arc tilt radian.
* start_point: the provided starting pose for end-effector.
