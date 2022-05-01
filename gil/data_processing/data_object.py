from gil.lgp.logic.action import Action, DurativeAction
import numpy as np
class Expert_motion_data(object):
    def __init__(self, action: DurativeAction, observation_task, observation_motion=None, command = None):
        self.action = action
        self.observation_task = observation_task
        self.observation_motion = observation_motion
        self.command = command

    def __str__(self):
        return self.action.__str__() + \
            "\nObservation" + str([str(obs) for obs in self.observation_list]) + \
            "\nCommand" + str([str(cmd) for cmd in self.command_list])
class Expert_task_data(object):
    def __init__(self, env_name:str, domain, problem, motion_data_list = [], data_tag = "", is_task_fully_refined=False):
        self.env_name = env_name
        self.domain = domain
        self.problem = problem
        self.data_tag = data_tag
        self.is_task_fully_refined = is_task_fully_refined
        self.motion_data_list = motion_data_list
    def add_motion_data(self, motion_data: Expert_motion_data):
        self.motion_data_list.append(motion_data)
