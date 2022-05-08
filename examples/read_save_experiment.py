import pickle
import os
from os.path import dirname, realpath, join, expanduser
EXPERIMENT_FILE_NAME = "set_table_with_holding_encoded_final"
PATH_FILE = dirname(realpath(__file__))
DATA_FOLDER = join(PATH_FILE, '../data', 'experiments')

FILE_NAME = join(DATA_FOLDER, EXPERIMENT_FILE_NAME)

FULL_EXPERIMENT = True
import numpy as np
with open(FILE_NAME, "rb") as f:
    exp_data = pickle.load(f)

#segment = exp_data["(\'p1_2\', 51396, 54791)"]
#print(segment)
total_segment = len(exp_data)
success_segment = 0.0
planning_time = []
success_planning_time = []
fail_planning_time = []
finish_percentage = []
finish_percentage_non_success = []

for segment, data in exp_data.items():
    if data["gil_success"]:
        success_segment+=1.0
        success_planning_time.append(data["gil_complete_time"])
    else:
        finish_percentage_non_success.append(data["gil_finish_percentage"])
        fail_planning_time.append(data['gil_complete_time'])
    planning_time.append(data["gil_complete_time"])
    finish_percentage.append(data["gil_finish_percentage"])
print("Total segment", total_segment)
print("GIL Success rate: ", success_segment/total_segment)
print("GIL Average planning time: {} +- {}".format(np.mean(planning_time),np.std(planning_time)))
print("GIL Average planning time for successful segment: {} +- {}".format(np.mean(success_planning_time),np.std(success_planning_time)))
print("GIL Average planning time for failed segment: {} +- {}".format(np.mean(fail_planning_time),np.std(fail_planning_time)))
print("GIL Average finish percentage: {} +- {}".format(np.mean(finish_percentage),np.std(finish_percentage)))
print("GIL Average finish percentage of non success segment: {} +- {}".format(np.mean(finish_percentage_non_success),np.std(finish_percentage_non_success)))
if FULL_EXPERIMENT:
    single_planning_time = []
    single_time = []
    single_success = 0.0
    dynamic_planning_time = []
    dynamic_success = 0.0
    dynamic_time = []
    for segment, data in exp_data.items():
        if data["single_success"]:
            single_success+=1.0
        if data["dynamic_success"]:
            dynamic_success+=1.0

        single_time.append(data["single_time"])
        dynamic_time.append(data["dynamic_time"])
    print("Success rate of single planning: ",single_success/total_segment)
    print("Average time of single planning: {} +- {}".format(np.mean(single_time),np.std(single_time)))
    print("Success rate of dynamic planning: ",dynamic_success/total_segment)
    print("Average time of dynamic planning: {} +- {}".format(np.mean(dynamic_time),np.std(dynamic_time)))
    pass