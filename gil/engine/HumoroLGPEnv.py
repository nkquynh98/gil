from gil.lgp.logic.action import DurativeAction
from gil.lgp.logic.problem import Problem
from gil.lgp.logic.domain import Domain
import numpy as np
import pybullet as p
from os.path import join, dirname, abspath, expanduser
import os
import time
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from gil.lgp.logic.parser import PDDLParser
from gil.lgp.core.planner import HumoroLGP
from gil.lgp.geometry.geometry import get_angle, get_point_on_circle
from gil.lgp.geometry.workspace import Circle
from gil.lgp.core.dynamic import HumoroDynamicLGP,DynamicLGP
from gil.lgp.geometry.workspace import HumoroWorkspace
from gil.lgp.core.planner import LGP, HumoroLGP
from gil.lgp.experiment.pipeline import Experiment
from gil.data_processing.data_process import encode_goal, encode_geometric_state, encode_action, process_recorded_data, decode_action
from gil.data_processing.data_object import Expert_task_data, Expert_motion_data
from gil.policy.model.MotionNet import *
from gil.policy.model.TaskNet import *
from gil.lgp.utils.helpers import *
import numpy as np

ROOT_DIR = join(dirname(abspath(__file__)), '../..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')

domain_file = join(DATA_DIR, 'domain_set_table.pddl')
robot_model_file = join(MODEL_DIR, 'pepper.urdf')
from examples.prediction.hmp_interface import HumanRollout
from gil.engine.HumoroLGPInteraction import HumoroLGPInteractionEnv
from humoro.utility import storeimg

class EnvHumoroLGP:
    '''
    Humoro environment to interfacing with humoro
    '''
    logger = logging.getLogger(__name__)
         
    def __init__(self, **kwargs):
        self.domain = PDDLParser.parse_domain(domain_file)
        path_to_mogaze = DATASET_DIR
        self.env_name = kwargs.get('env_name', "default_env")
        self.sim_fps = kwargs.get('sim_fps', 120)
        self.prediction = kwargs.get('prediction', False)
        self.save_training_data = kwargs.get('save_training_data', False)
        #self.hr=None
        self.hr = HumanRollout(path_to_mogaze=path_to_mogaze, fps=self.sim_fps, predicting=self.prediction, load_predictions=True)
        self.humoro_lgp = HumoroLGPInteractionEnv(self.domain, self.hr, robot_model_file=robot_model_file,**kwargs)
        #print("Box extent 2",self.humoro_lgp.workspace.box.box_extent())
        # useful variables
        self.robot_frame = self.humoro_lgp.workspace.robot_frame
        self.handling_circle = Circle(np.zeros(2), radius=0.3)
        self.reset_experiment()

    def step(self, vel):
        self.humoro_lgp.command_vel(vel)

    def get_problem(self, segment):
        self.problem = None
        self.segment = segment
        start_agent_symbols = frozenset([('agent-avoid-human',), ('agent-free',)])
        end_agent_symbols = frozenset([('agent-at', 'table')])
        objects = self.hr.get_object_carries(segment, predicting=False)
        self.problem = Experiment.get_problem_from_segment(self.hr, segment, self.domain, objects, start_agent_symbols, end_agent_symbols)
        self.problem_encoded = encode_goal(self.problem)
        return self.problem 

    

    def init_env(self, **kwargs):
        self.humoro_lgp.init_env(**kwargs)
        self.prev_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
        self.q = [0, 0, 0, 1]
        self.z_angle = 0.0
        self.workspace = self.humoro_lgp.workspace
        self.actual_robot_path = []
        self.actual_human_path = []

    def init_planner(self, **kwargs):
        if 'problem' not in kwargs:
            kwargs['problem'] = self.problem
        kwargs['sim_fps'] = self.sim_fps
        kwargs['prediction'] = self.prediction
        self.humoro_lgp.init_planner(**kwargs)
        self.prev_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
        self.save_training_data = kwargs.get('save_training_data', False)
        data_tag = kwargs.get('data_tag', "")
        if self.save_training_data:
            self.expert_data = Expert_task_data(self.env_name, self.domain, self.humoro_lgp.problem, data_tag=data_tag)
        self.problem_encoded = encode_goal(self.humoro_lgp.problem)
        self.q = [0, 0, 0, 1]
        self.z_angle = 0.
        self.actual_robot_path = []
        self.actual_human_path = []

    def reset_experiment(self):
        # single plan
        self.single_symbolic_plan_time = 0
        self.single_plans = []
        self.single_chosen_plan_id = None
        self.single_perceive_human_objects = []
        self.single_geometric_plan_time = 0
        self.single_plan_costs = []
        self.single_num_failed_plan = 0
        self.single_actual_path = None
        self.single_complete_time = 0
        self.single_reduction_ratio = 0.
        # dynamic plan
        self.dynamic_symbolic_plan_time = {}
        self.dynamic_plans = {}
        self.dynamic_chosen_plan_id = {}
        self.dynamic_perceive_human_objects = {}
        self.dynamic_geometric_plan_time = {}
        self.dynamic_plan_costs = {}
        self.dynamic_num_failed_plans = {}
        self.dynamic_num_change_plan = 0
        self.dynamic_actual_path = None
        self.dynamic_complete_time = 0
        self.dynamic_reduction_ratio = 0.
    
    def get_experiment_data(self):
        data = {
            'single_symbolic_plan_time': self.single_symbolic_plan_time,
            'single_plans': self.single_plans,
            'single_chosen_plan_id': self.single_chosen_plan_id,
            'single_perceive_human_objects': self.single_perceive_human_objects,
            'single_geometric_plan_time': self.single_geometric_plan_time,
            'single_plan_costs': self.single_plan_costs,
            'single_num_failed_plan': self.single_num_failed_plan,
            'single_actual_path': self.single_actual_path,
            'single_complete_time': self.single_complete_time,
            'single_reduction_ratio': self.single_reduction_ratio,
            'dynamic_symbolic_plan_time': self.dynamic_symbolic_plan_time,
            'dynamic_plans': self.dynamic_plans,
            'dynamic_chosen_plan_id': self.dynamic_chosen_plan_id,
            'dynamic_perceive_human_objects': self.dynamic_perceive_human_objects,
            'dynamic_geometric_plan_time': self.dynamic_geometric_plan_time,
            'dynamic_plan_costs': self.dynamic_plan_costs,
            'dynamic_num_failed_plans': self.dynamic_num_failed_plans,
            'dynamic_num_change_plan': self.dynamic_num_change_plan,
            'dynamic_actual_path': self.dynamic_actual_path,
            'dynamic_complete_time': self.dynamic_complete_time,
            'dynamic_reduction_ratio': self.dynamic_reduction_ratio,
            'human_path': self.actual_human_path
        }
        return data

    def check_goal_reached(self):
        return self.humoro_lgp.logic_planner.current_state in self.humoro_lgp.logic_planner.goal_states

    def update_visualization(self):
        '''
        This update currently has no playback (backward in time)
        '''
        # update robot
        robot = self.humoro_lgp.workspace.get_robot_link_obj()
        current_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
        #print("Current pos", current_robot_pos)
        #print("robot", robot)
        grad = current_robot_pos - self.prev_robot_pos
        if np.linalg.norm(grad) > 0:  # prevent numerical error
            z_angle = get_angle(grad, np.array([1, 0]))  # angle of current path gradient with y axis
            self.z_angle = z_angle if grad[1] > 0 else -z_angle
            self.q = p.getQuaternionFromEuler([0, 0, self.z_angle])  # + pi/2 due to default orientation of pepper is x-axis
        self.prev_robot_pos = current_robot_pos
        p.resetBasePositionAndOrientation(self.humoro_lgp.player._robots[self.robot_frame], [*current_robot_pos, 0], self.q)
        # update object
        if self.humoro_lgp.plan is not None:
            current_action = self.humoro_lgp.get_current_action()
            if current_action is not None and current_action.name == 'place':
                obj, location = current_action.parameters
                box = self.humoro_lgp.workspace.kin_tree.nodes[location]['link_obj']
                x = np.random.uniform(box.origin[0] - box.dim[0] / 2, box.origin[0] + box.dim[0] / 2)  # TODO: should be desired place_pos on location, or add an animation of placing here
                y = np.random.uniform(box.origin[1] - box.dim[1] / 2, box.origin[1] + box.dim[1] / 2)
                p.resetBasePositionAndOrientation(self.humoro_lgp.player._objects[obj], [x, y, 0.735], [0, 0, 0, 1])  # currently ignore object orientation
            elif robot.couplings:
                for obj in robot.couplings:
                    self.handling_circle.origin = current_robot_pos
                    handling_pos = get_point_on_circle(self.z_angle, self.handling_circle)
                    p.resetBasePositionAndOrientation(self.humoro_lgp.player._objects[obj], [*handling_pos, 1], [0, 0, 0, 1])  # TODO: for now attach object at robot origin

    def get_current_robot_velocity(self):
        current_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
        velocity = current_robot_pos - self.prev_robot_pos
        return velocity
    def run(self, replan=False, sleep=False, save_frame=False):
        if not replan:
            self.humoro_lgp.update_current_symbolic_state()
            start_symbolic_plan = time.time()
            success = self.humoro_lgp.symbolic_plan()
            start_geometric_plan = time.time()
            self.single_symbolic_plan_time = start_geometric_plan - start_symbolic_plan
            success = self.humoro_lgp.geometric_plan()
            self.single_geometric_plan_time = time.time() - start_geometric_plan
            self.single_plans = self.humoro_lgp.get_list_plan_as_string()
            #print("List plan: ", self.single_plans)
            self.single_chosen_plan_id = self.humoro_lgp.chosen_plan_id
            self.single_perceive_human_objects = self.humoro_lgp.perceive_human_objects
            self.single_plan_costs = self.humoro_lgp.ranking
            for r in self.humoro_lgp.ranking:
                if r[1] == self.humoro_lgp.chosen_plan_id:
                    break
                self.single_num_failed_plan += 1
            if not success:
                HumoroDynamicLGP.logger.info('Task failed!')
                return False
        max_t = self.humoro_lgp.timeout * self.humoro_lgp.ratio
        #Added
        #self.humoro_lgp.workspace.draw_workspace()
        #self.humoro_lgp.workspace.draw_kinematic_tree()
        #self.humoro_lgp.draw_potential_heightmap()
        #self.humoro_lgp.workspace.draw_kinematic_tree()
        #self.humoro_lgp.workspace.draw_workspace()
        #Added
        while self.humoro_lgp.lgp_t < max_t:
            if replan and (self.humoro_lgp.lgp_t % (self.humoro_lgp.trigger_period * self.humoro_lgp.ratio) == 0):
                self.humoro_lgp.update_current_symbolic_state()
                if self.humoro_lgp.plan is None:
                    self.dynamic_num_change_plan += 1
                    start_symbolic_plan = time.time()
                    success = self.humoro_lgp.symbolic_plan()
                    self.dynamic_symbolic_plan_time[self.humoro_lgp.lgp_t] = time.time() - start_symbolic_plan
                    self.dynamic_perceive_human_objects[self.humoro_lgp.lgp_t] = self.humoro_lgp.perceive_human_objects
                start_geometric_plan = time.time()
                success = self.humoro_lgp.geometric_replan()
                self.dynamic_geometric_plan_time[self.humoro_lgp.lgp_t] = time.time() - start_geometric_plan
                if self.humoro_lgp.lgp_t in self.dynamic_symbolic_plan_time:
                    self.dynamic_chosen_plan_id[self.humoro_lgp.lgp_t] = self.humoro_lgp.chosen_plan_id
                    self.dynamic_plans[self.humoro_lgp.lgp_t] = self.humoro_lgp.get_list_plan_as_string()
                    self.dynamic_plan_costs[self.humoro_lgp.lgp_t] = self.humoro_lgp.ranking
                    if success:
                        n = 0
                        for r in self.humoro_lgp.ranking:
                            if r[1] == self.humoro_lgp.chosen_plan_id:
                                break
                            n += 1
                        self.dynamic_num_failed_plans[self.humoro_lgp.lgp_t] = n
                    else:
                        self.dynamic_num_failed_plans[self.humoro_lgp.lgp_t] = len(self.humoro_lgp.ranking)
            if self.humoro_lgp.lgp_t % self.humoro_lgp.ratio == 0:
                # executing current action in the plan
                if replan:
                    if success:
                        self.humoro_lgp.act(sanity_check=False)
                else:
                    self.humoro_lgp.act(sanity_check=False)
                
                # reflecting changes in PyBullet
                if self.save_training_data:
                    self.get_data_for_training()  
                self.update_visualization()
  
                # recording paths
                self.actual_robot_path.append(self.humoro_lgp.workspace.get_robot_geometric_state())
                self.actual_human_path.append(self.humoro_lgp.workspace.get_human_geometric_state())
            self.humoro_lgp.update_workspace()
            self.humoro_lgp.visualize()
            if save_frame:
                storeimg(p, os.path.join(self.image_dir, str(self.humoro_lgp.lgp_t) + '.png'))
            self.humoro_lgp.increase_timestep()
            if self.humoro_lgp.lgp_t > self.humoro_lgp.workspace.duration and self.humoro_lgp.symbolic_elapsed_t > self.humoro_lgp.get_current_plan_time():
                break
            if sleep:
                time.sleep(1 / self.humoro_lgp.sim_fps)
        self.humoro_lgp.update_workspace()
        self.humoro_lgp.update_current_symbolic_state()
        if not replan:
            self.single_actual_path = self.actual_robot_path
            self.single_complete_time = self.humoro_lgp.lgp_t / self.humoro_lgp.sim_fps
            self.single_reduction_ratio = self.humoro_lgp.lgp_t / self.hr.get_segment_timesteps(self.humoro_lgp.workspace.segment, predicting=False)
        else:
            self.dynamic_actual_path = self.actual_robot_path
            self.dynamic_complete_time = self.humoro_lgp.lgp_t / self.humoro_lgp.sim_fps
            self.dynamic_reduction_ratio = self.humoro_lgp.lgp_t / self.hr.get_segment_timesteps(self.humoro_lgp.workspace.segment, predicting=False)
        if self.check_goal_reached():
            HumoroDynamicLGP.logger.info('Task complete successfully!')
            if self.save_training_data:
                process_recorded_data(self.expert_data)
            return True
        else:
            HumoroDynamicLGP.logger.info('Task failed!')
            return False


    def get_data_for_training(self):
        current_action = self.humoro_lgp.get_current_action()
        if current_action is not None:
            obs_task = self.humoro_lgp.get_observation_for_task()
            obs_motion = self.humoro_lgp.get_observation_for_motion(current_action)
            command = None
            if current_action.name == "move":
                command = self.get_current_robot_velocity()

            print("command", command)
            temp_motion_data = Expert_motion_data(current_action,obs_task,obs_motion,command)
            self.expert_data.add_motion_data(temp_motion_data)

    # For testing the trained network
    def load_trained_network(self, json_config_link:str):
        self.is_network_loaded = True
        with open(json_config_link,"r") as f:
            config = json.load(f)
        #Create the net instance based on a name (string)
        MODEL_FOLDER = config["saved_model_folder"]
        param = config["tasknet"]["parameters"]
        self.tasknet = eval(config["tasknet"]["type"])(param["goal_encoded_dim"], param["observation_dim"], param["action_dim"], param["object_dim"], param["location_dim"])
        #Load network
        self.tasknet.load_state_dict(torch.load(MODEL_FOLDER+config["tasknet"]["file_name"]))
        self.tasknet.eval()
        self.motion_net = {}

        for action_name, action_net in config["motionnet"].items():
            param = action_net["parameters"]
            self.motion_net[action_name] = eval(action_net["type"])(param["observation_dim"],param["action_dim"])
            self.motion_net[action_name].load_state_dict(torch.load(MODEL_FOLDER+action_net["file_name"]))
            self.motion_net[action_name].eval()



    def run_gil(self,sleep=False):
        # For recording data
        self.gil_robot_path = []
        self.actual_human_path = []
        assert self.is_network_loaded
        max_t = self.humoro_lgp.timeout * self.humoro_lgp.ratio
        while self.humoro_lgp.lgp_t < max_t:
            if self.humoro_lgp.lgp_t % self.humoro_lgp.ratio == 0:
                action_name, location, object = self.get_tasknet_output()
                # action
                objects = {"location": [location], "object": [object]}
                predict_action = self.domain.ground_single_action(action_name, objects)
                print(predict_action)
                # Move or pick/place
                if action_name == "move":
                    command = self.get_motionnet_output(predict_action).numpy()
                    self.humoro_lgp.command_vel(command)
                else:
                    self.humoro_lgp.action_map[action_name](predict_action)

                #self.humoro_lgp.act(sanity_check=False)
                
                # reflecting changes in PyBullet
                if self.save_training_data:
                    self.get_data_for_training()  
                self.update_visualization()
                self.gil_robot_path.append(self.humoro_lgp.workspace.get_robot_geometric_state())
                self.actual_human_path.append(self.humoro_lgp.workspace.get_human_geometric_state())
                
            self.humoro_lgp.update_workspace()
            self.humoro_lgp.visualize()
            self.humoro_lgp.increase_timestep()
            #time.sleep(0.1)
            if self.humoro_lgp.lgp_t > self.humoro_lgp.workspace.duration and self.humoro_lgp.symbolic_elapsed_t > self.humoro_lgp.get_current_plan_time():
                break
            if sleep:
                time.sleep(1 / self.humoro_lgp.sim_fps)
        self.humoro_lgp.update_workspace()
        self.humoro_lgp.update_current_symbolic_state()
        if self.check_goal_reached():
            HumoroDynamicLGP.logger.info('Task complete successfully!')
            if self.save_training_data:
                process_recorded_data(self.expert_data)
            return True
        else:
            HumoroDynamicLGP.logger.info('Task failed!')
            return False


    def get_tasknet_output(self):
        assert self.is_network_loaded
        observation = self.humoro_lgp.get_observation_for_task()
        input_task = np.concatenate([self.problem_encoded,observation])
        with torch.no_grad():
            action_encoded, location_encoded, object_encoded = self.tasknet(torch.from_numpy(input_task).float())
        
        action_encoded = action_encoded.numpy()
        object_encoded = object_encoded.numpy()
        location_encoded = location_encoded.numpy()

        action_decoded, location_decoded, object_decoded = decode_action(action_encoded, location_encoded, object_encoded)
        return action_decoded, location_decoded, object_decoded
    def get_motionnet_output(self, action:DurativeAction):
        observation = self.humoro_lgp.get_observation_for_motion(action)
        #Get the action command
        with torch.no_grad():
            command = self.motion_net[action.name].forward(torch.from_numpy(observation).float())
        return command 


    def draw_real_path(self, show=False, human=False, gil=False, planned = False, save_file:str = None):
        ax = self.humoro_lgp.workspace.draw_workspace(False)
        if gil:
            draw_numpy_trajectory(ax,self.gil_robot_path,"b")
        if human:
            draw_numpy_trajectory(ax,self.actual_human_path,"r")
        if planned:
            draw_numpy_trajectory(ax,self.actual_robot_path,"g")
        if show:
            plt.show()
        if save_file is not None:
            plt.savefig(fname=save_file)