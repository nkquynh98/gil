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
from gil.lgp.core.dynamic import HumoroDynamicLGP
import numpy as np

ROOT_DIR = join(dirname(abspath(__file__)), '../..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')

domain_file = join(DATA_DIR, 'domain_set_table.pddl')
robot_model_file = join(MODEL_DIR, 'pepper.urdf')
from examples.prediction.hmp_interface import HumanRollout
from humoro.utility import storeimg
        
class EnvHumoroLGP(HumoroDynamicLGP):
    def __init__(self, **kwargs):
        super(EnvHumoroLGP, self).__init__(domain_file=domain_file, robot_model_file=robot_model_file, path_to_mogaze=DATASET_DIR,**kwargs)
        
    
    def init_env(self):
        pass
    
    def step(self):
        pass
    def get_observation(self):
        pass


 