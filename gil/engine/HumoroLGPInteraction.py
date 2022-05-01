from threading import local
from gil.lgp.logic.problem import Problem
from gil.lgp.logic.domain import Domain
import numpy as np
import pybullet as p
from os.path import join, dirname, abspath, expanduser
import os
import time
from datetime import datetime
import logging
from gil.lgp.logic.parser import PDDLParser
from gil.lgp.core.planner import HumoroLGP
from gil.lgp.geometry.geometry import get_angle, get_point_on_circle
from gil.lgp.geometry.workspace import Circle
from gil.lgp.core.dynamic import HumoroDynamicLGP,DynamicLGP
from gil.lgp.geometry.workspace import HumoroWorkspace
from gil.lgp.core.planner import LGP, HumoroLGP
from gil.lgp.experiment.pipeline import Experiment
from gil.data_processing.data_process import encode_goal, encode_geometric_state
from gil.lgp.logic.action import DurativeAction, Action
import numpy as np

ROOT_DIR = join(dirname(abspath(__file__)), '../..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')

domain_file = join(DATA_DIR, 'domain_set_table.pddl')
robot_model_file = join(MODEL_DIR, 'pepper.urdf')
from examples.prediction.hmp_interface import HumanRollout
from humoro.utility import storeimg

class HumoroLGPInteractionEnv(HumoroLGP):
    def __init__(self, domain, hr, **kwargs):
        super().__init__(domain, hr, **kwargs)

    def init_env(self, **kwargs):
        # workspace
        segment = tuple(kwargs.get('segment'))
        human_carry = kwargs.get('human_carry', 0)
        prediction = kwargs.get('prediction', False)
        # logic planner params
        problem = kwargs.get('problem')
        self.workspace.initialize_workspace_from_humoro(segment=segment, human_carry=human_carry, prediction=prediction, objects=problem.objects['object'])
    
    
    def init_planner(self, **kwargs):
        # LGP params
        self.trigger_period = kwargs.get('trigger_period', 10)  # with fps
        self.timeout = kwargs.get('timeout', 1000)  # fps timesteps
        self.sim_fps = kwargs.get('sim_fps', 120)  # simulation fps
        self.fps = kwargs.get('fps', 10)  # sampling fps
        self.human_freq = kwargs.get('human_freq', 40)  # human placement frequency according to fps
        self.traj_init = kwargs.get('traj_init', 'outer')  # initialization scheme for trajectory
        self.window_len = kwargs.get('window_len', 'max')  # frames, according to this sampling fps
        self.full_replan = kwargs.get('full_replan', True)
        self.ratio = int(self.sim_fps / self.fps)
        # logic planner params
        self.problem = kwargs.get('problem')
        #if self.problem is not None:
        #    self.encoded_goal = encode_goal(problem=problem)
        ignore_cache = kwargs.get('ignore_cache', False)
        # workspace
        segment = tuple(kwargs.get('segment'))
        human_carry = kwargs.get('human_carry', 0)
        prediction = kwargs.get('prediction', False)
        # init components
        self.logic_planner.init_planner(problem=self.problem, ignore_cache=ignore_cache)
        self.workspace.initialize_workspace_from_humoro(segment=segment, human_carry=human_carry, prediction=prediction, objects=self.problem.objects['object'])
        if self.window_len == 'max':
            self.window_len = int(self.workspace.duration / self.ratio)
        init_symbols = self.symbol_sanity_check()
        constant_symbols = [p for p in init_symbols if p[0] not in self.workspace.DEDUCED_PREDICATES]
        self.workspace.set_constant_symbol(constant_symbols)
        self._precompute_human_placement()
        self.landmarks = {
            'table': get_point_on_circle(np.pi/2, self.workspace.kin_tree.nodes['table']['limit']),
            'big_shelf': get_point_on_circle(np.pi, self.workspace.kin_tree.nodes['big_shelf']['limit']),
            'small_shelf': get_point_on_circle(0, self.workspace.kin_tree.nodes['small_shelf']['limit'])
        }
        # dynamic parameters
        self.reset()
   
    def command_vel(self, vel:np.ndarray):
        assert vel.shape[0] == 2
        # Velocity in x_vel, y_vel
        current_robot_pose = self.workspace.get_robot_geometric_state()
        current_robot_pose += vel
        self.workspace.set_robot_geometric_state(current_robot_pose)
        #print(current_robot_pose)

    def get_observation_for_task(self, with_lidar = False):
        geometric_state = self.workspace.geometric_state
        # Encoded goal + current geometry state + lidar(optional) + vector_to_target
        encoded_geometry = encode_geometric_state(geometric_state)
        # encoded_goal = self.encoded_goal
        # current_observation = np.concatenate([encoded_goal,encoded_geometry])
        current_observation = encoded_geometry
        if with_lidar:
            lidar_scan_value = self.workspace.get_lidar_result2D(numRays=10)
            current_observation = np.concatenate([current_observation, lidar_scan_value])

        return current_observation

    def get_observation_for_motion(self, action:DurativeAction):
        geometry_state = self.workspace.geometric_state
        robot_pose = self.workspace.get_robot_geometric_state()
        lidar_scan_value = self.workspace.get_lidar_result2D(numRays=10)
        observation = np.concatenate([robot_pose,lidar_scan_value])
        if action.name == "move":
            location = action.parameters[0]
            distance = geometry_state[location]-robot_pose
            observation = np.concatenate([observation,distance])
            return observation
        else:
            return None

    def get_current_velocity(self):
        pass





    
