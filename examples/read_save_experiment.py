import pickle
import os
from os.path import dirname, realpath, join, expanduser
EXPERIMENT_FILE_NAME = "set_table_with_holding_encoded"
PATH_FILE = dirname(realpath(__file__))
DATA_FOLDER = join(PATH_FILE, '../data', 'experiments')

FILE_NAME = join(DATA_FOLDER, EXPERIMENT_FILE_NAME)
import numpy as np
with open(FILE_NAME, "rb") as f:
    exp_data = pickle.load(f)

#segment = exp_data["(\'p1_2\', 51396, 54791)"]
#print(segment)
total_segment = len(exp_data)
success_segment = 0.0
planning_time = []
success_planning_time = []
finish_percentage = []
finish_percentage_non_success = []
for segment, data in exp_data.items():
    if data["gil_success"]:
        success_segment+=1.0
        success_planning_time.append(data["gil_complete_time"])
    else:
        finish_percentage_non_success.append(data["gil_finish_percentage"])
    planning_time.append(data["gil_complete_time"])
    finish_percentage.append(data["gil_finish_percentage"])
print("Total segment", total_segment)
print("Success rate: ", success_segment/total_segment)
print("Average planning time: {} +- {}".format(np.mean(planning_time),np.std(planning_time)))
print("Average planning time for success segment: {} +- {}".format(np.mean(success_planning_time),np.std(success_planning_time)))
print("Average finish percentage: {} +- {}".format(np.mean(finish_percentage),np.std(finish_percentage)))
print("Average finish percentage of non success segment: {} +- {}".format(np.mean(finish_percentage_non_success),np.std(finish_percentage_non_success)))