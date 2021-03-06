import sys
import argparse
import time
import yaml
import logging
from ast import literal_eval as make_tuple
from os.path import join, dirname, abspath, expanduser
import os
import matplotlib
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

logging.basicConfig(level=logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
sys.path.append(ROOT_DIR)
from gil.lgp.experiment.pipeline import Experiment
from gil.engine.HumoroLGPEnv import EnvHumoroLGP

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_lgp.py --segment \"(\'p7_3\', 29439, 33249)\"')
parser.add_argument('--segment', help='The scenario name of the domain, problem file', type=str, default="(\'p5_1\', 100648, 108344)")
parser.add_argument('-d', help='dynamic', type=bool, default=False)
parser.add_argument('-p', help='prediction', type=bool, default=False)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

segment = make_tuple(args.segment)
sim_fps = 30 if args.p else 120 
env = EnvHumoroLGP(sim_fps=sim_fps, prediction=args.p, enable_viewer=True, verbose=args.v)
problem = env.get_problem(segment=segment)
print(problem)
env.init_env(segment=segment, problem=problem, human_freq='human-at', traj_init='outer')
#env.humoro_lgp.update_current_symbolic_state()
#env.humoro_lgp.symbolic_plan()
#success = env.humoro_lgp.geometric_plan()
# while True:
#     env.humoro_lgp.act()
# 
#     env.update_visualization()
shelf_pos = env.humoro_lgp.workspace._geometric_state["small_shelf"]
while True:
    vel = 0.001
    robot_pos= env.humoro_lgp.workspace.get_robot_geometric_state()
    command = vel*np.array((shelf_pos-robot_pos))
    env.step(command)
    env.update_visualization()
    #env.humoro_lgp.get_current_observation()
    #print(env.humoro_lgp.workspace.get_lidar_result2D(numRays=37))
    #print(env.humoro_lgp.workspace._geometric_state)
    env.humoro_lgp.workspace.update_symbolic_state()
    #print(env.humoro_lgp.workspace._symbolic_state)
    time.sleep(0.01)
    