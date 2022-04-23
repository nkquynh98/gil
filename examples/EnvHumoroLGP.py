import sys
import argparse
import time
import yaml
import logging
from ast import literal_eval as make_tuple
from os.path import join, dirname, abspath, expanduser
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

logging.basicConfig(level=logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
sys.path.append(ROOT_DIR)
from gil.lgp.experiment.pipeline import Experiment
from gil.engine.LGP_engine import EnvHumoroLGP

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
start_agent_symbols = frozenset([('agent-avoid-human',), ('agent-free',)])
end_agent_symbols = frozenset([('agent-at', 'table')])
objects = env.hr.get_object_carries(segment, predicting=False)
problem = Experiment.get_problem_from_segment(env.hr, segment, env.domain, objects, start_agent_symbols, end_agent_symbols)
env.init_planner(segment=segment, problem=problem, human_freq='human-at', traj_init='outer')
env.humoro_lgp.update_current_symbolic_state()
    
#env.humoro_lgp.symbolic_plan()
#success = env.humoro_lgp.geometric_plan()
# while True:
#     env.humoro_lgp.act()
#     env.update_visualization()
print(objects)
