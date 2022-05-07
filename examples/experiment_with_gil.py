import sys
import argparse
import time
import yaml
import logging
from datetime import datetime
from os.path import join, dirname, abspath, expanduser
from gil.engine.GIL_dataset_generator import GILDatasetGenerator
ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
JSON_CONFIG = join(ROOT_DIR, 'data', 'configuration', "policy_configuration_2.json")

DATASET_NAME = "with_holding_encoded"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', help='The scenario name of the domain, problem file', type=str, default='set_table')
parser.add_argument('-c', help='number of human carry', type=int, default=3)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

start_time = time.time()
experiment = GILDatasetGenerator(dataset_name = DATASET_NAME, task=args.s, human_carry=args.c, verbose=args.v)
init_time = time.time()
print('Init time: ' + str(init_time - start_time) + 's')
experiment.run_with_gil(json_config_file=JSON_CONFIG, single_plan=False, dynamic_plan=False, save_fig=True)
experiment.save_data()