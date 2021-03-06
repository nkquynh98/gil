import numpy as np
import pybullet as p
import time
from datetime import datetime
import logging
from gil.lgp.logic.parser import PDDLParser
from gil.lgp.core.planner import HumoroLGP
from gil.lgp.geometry.geometry import get_angle, get_point_on_circle
from gil.lgp.geometry.workspace import Circle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# temporary importing until complication of install is resolve
import os
import sys
_path_file = os.path.dirname(os.path.realpath(__file__))
VIDEO_DIR = os.path.join(_path_file, '../../data/videos')
sys.path.append(os.path.join(_path_file, "../../../humoro"))
sys.path.append("../../../../")
from examples.prediction.hmp_interface import HumanRollout
from humoro.utility import storeimg


class DynamicLGP(object):
    '''
    General dynamic LGP class
    '''
    def __init__(self, **kwargs):
        domain_file = kwargs.get('domain_file')
        problem_file = kwargs.get('problem_file', None)
        self.domain = PDDLParser.parse_domain(domain_file)
        self.problem = None
        if problem_file is not None:
            self.problem = PDDLParser.parse_problem(problem_file)
    
    def run(self):
        raise NotImplementedError()


class HumoroDynamicLGP(DynamicLGP):
    '''
    Humoro environment to interfacing with humoro
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        super(HumoroDynamicLGP, self).__init__(**kwargs)
        path_to_mogaze = kwargs.get('path_to_mogaze', 'datasets/mogaze')
        self.sim_fps = kwargs.get('sim_fps', 120)
        self.prediction = kwargs.get('prediction', False)
        #self.hr=None
        self.hr = HumanRollout(path_to_mogaze=path_to_mogaze, fps=self.sim_fps, predicting=self.prediction, load_predictions=True)
        self.humoro_lgp = HumoroLGP(self.domain, self.hr, **kwargs)
        #print("Box extent 2",self.humoro_lgp.workspace.box.box_extent())
        # useful variables
        self.robot_frame = self.humoro_lgp.workspace.robot_frame
        self.handling_circle = Circle(np.zeros(2), radius=0.3)
        self.reset_experiment()
        self.image_dir = os.path.join(VIDEO_DIR, str(datetime.now()))
        os.makedirs(self.image_dir, exist_ok=True)
    
    def init_planner(self, **kwargs):
        if 'problem' not in kwargs:
            kwargs['problem'] = self.problem
        kwargs['sim_fps'] = self.sim_fps
        kwargs['prediction'] = self.prediction
        self.humoro_lgp.init_planner(**kwargs)
        self.prev_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
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
            return True
        else:
            HumoroDynamicLGP.logger.info('Task failed!')
            return False
            