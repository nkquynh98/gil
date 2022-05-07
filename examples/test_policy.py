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
JSON_CONFIG = join(ROOT_DIR, 'data', 'configuration', "policy_configuration_2.json")

sys.path.append(ROOT_DIR)
from gil.lgp.experiment.pipeline import Experiment
from gil.engine.HumoroLGPEnv import EnvHumoroLGP

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_lgp.py --segment \"(\'p7_3\', 29439, 33249)\"')
parser.add_argument('--segment', help='The scenario name of the domain, problem file', type=str, default="(\'p7_3\', 136064, 138364)")
parser.add_argument('-d', help='dynamic', type=bool, default=False)
parser.add_argument('-p', help='prediction', type=bool, default=False)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

segment = make_tuple(args.segment)
sim_fps = 30 if args.p else 120 
env = EnvHumoroLGP(sim_fps=sim_fps, prediction=args.p, enable_viewer=True, verbose=args.v)
problem = env.get_problem(segment=segment)
env.init_planner(segment=segment, problem=problem, human_carry=3, trigger_period=10, human_freq='human-at', traj_init='outer')
#env.humoro_lgp.update_current_symbolic_state()
#env.humoro_lgp.symbolic_plan()
#success = env.humoro_lgp.geometric_plan()
# while True:
#     env.humoro_lgp.act()
# 
#     env.update_visualization()
#print(env.humoro_lgp.logic_planner.ground_actions)

env.load_trained_network(JSON_CONFIG)

result = env.run_gil()

#env.init_planner(segment=segment, problem=problem, human_carry=3, trigger_period=10, human_freq='human-at', traj_init='outer')
env.draw_real_path(gil=True, save_file="abc.png")
env.draw_real_path(save_file="abcd.png")
#env.run()
#env.draw_real_path(gil=True, show=True, human=True, planned=True)