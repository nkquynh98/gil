import sys
import argparse
import time
import yaml
import logging
from datetime import datetime
from gil.engine.GIL_dataset_generator import GILDatasetGenerator

DATASET_NAME = "without_human_"+str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', help='The scenario name of the domain, problem file', type=str, default='set_table')
parser.add_argument('-c', help='number of human carry', type=int, default=3)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

start_time = time.time()
experiment = GILDatasetGenerator(dataset_name = DATASET_NAME, task=args.s, human_carry=args.c, verbose=args.v)
init_time = time.time()
print('Init time: ' + str(init_time - start_time) + 's')
experiment.run(gather_data=True)
#experiment.save_data()