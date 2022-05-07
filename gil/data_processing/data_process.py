# Data processing for mogaze dataset
import time
from datetime import datetime
import h5py
import os
from gil.lgp.logic.problem import Problem
from gil.lgp.logic.domain import Domain
from gil.lgp.logic.action import Action 
from gil.data_processing.data_object import Expert_task_data
import numpy as np

LOCATION = ['table', 'small_shelf', 'big_shelf']
OBJECT = ['cup_red', 'cup_green', 'cup_blue', 'cup_pink', 'plate_pink', 'plate_red', 'plate_green', 'plate_blue', 'jug', 'bowl']
ACTION = ["move", "pick", "place"]
CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = CURRENT_FOLDER + "/../../datasets/gil_dataset/"
FILE_NAME = "example_task_data.pickle"
def encode_goal(problem: Problem):
    
    problem_dict = problem.get_dict()
    goal = problem_dict["positive_goal"][0]
    
    goal_encoded = np.zeros(len(OBJECT))
    for sub_goal in goal:
        if sub_goal[0] == "on" :    
            #print(sub_goal)
            # index of the encoded goal: if the goal is placed object 1 at target 3: goalencoded[1*5+3]=1
            index = OBJECT.index(sub_goal[1])
            goal_encoded[index] = 1
    return goal_encoded        


def encode_geometric_state(geometry: dict):
    robot_pose = geometry["robot"]
    table_pose = geometry["table"]
    big_shelf = geometry["big_shelf"]
    small_shelf = geometry["small_shelf"]
    OFFSET = 4
    encoded_value = np.zeros(OFFSET*2+len(OBJECT)*2)
    encoded_value[0:2] = robot_pose
    encoded_value[2:4] = table_pose-robot_pose
    encoded_value[4:6] = big_shelf-robot_pose
    encoded_value[6:8] = small_shelf-robot_pose

    for i, object in enumerate(OBJECT):
        if object in geometry:
            #Calculate the distance from the object to the target
            encoded_value[OFFSET*2+i*2:OFFSET*2+(i+1)*2] = table_pose - geometry[object]
    return encoded_value
    
def encode_robot_holding(objects):
    encoded_value = np.zeros(len(OBJECT))
    if len(objects)>0:
        for object in objects:
            index = OBJECT.index(object)
            encoded_value[index] = 1        
    return encoded_value    



def encode_logic_state(logic):
    pass

def encode_human_holding(logic):
    pass
def encode_action(action_to_encode: Action):
    action_encoded = np.zeros((len(ACTION)))
    object_encoded = np.zeros((len(OBJECT)))
    location_encoded = np.zeros((len(LOCATION)))

    for i, action in enumerate(ACTION):
        if action==action_to_encode.name:
            action_encoded[i] = 1
    
    if len(action_to_encode.parameters)==2:
        location_encoded[LOCATION.index(action_to_encode.parameters[1])]=1
        object_encoded[OBJECT.index(action_to_encode.parameters[0])]=1
    else:
        location_encoded[LOCATION.index(action_to_encode.parameters[0])]=1
    return [action_encoded, location_encoded, object_encoded]

def decode_action(encoded_action, encoded_location, encoded_object):
    for i, action in enumerate(ACTION):
        if i == np.argmax(encoded_action):
            decoded_action = action

    decoded_object = OBJECT[np.argmax(encoded_object)]
    decoded_location= LOCATION[np.argmax(encoded_location)]

    return decoded_action, decoded_location, decoded_object


def safe_open(path, rw:str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, rw)

def save_to_hdf5(file_name:str, data_dict: dict):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    h5_file = h5py.File(file_name, 'a')
    datalength = 0
    for dataset_name, dataset_value in data_dict.items():
        # Save each list of the data as an dataset (multi-dim np array) in the hdf5 file
        if len(dataset_value)>0:
            dataset_value = np.vstack(dataset_value)
            datalength = dataset_value.shape[0]
            h5_file.create_dataset(dataset_name, data=dataset_value, compression="gzip", chunks=True)
    h5_file.attrs["data_length"]=datalength
def process_recorded_data(data: Expert_task_data):
    env_name = data.env_name
    domain_name = data.domain.name
    #file_name = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))+".pickle"
    #file_name = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))+".h5"
    file_name = data.problem.name + "_"+data.data_tag+ ".h5"
    #processed_task_data = []
    #The prefix number is added to ensure the dict key appears in the desired orders
    task_data = {"0_observation": [], "1_action": [], "2_locations": [], "3_objects": [] }
    motion_data = {}
    #processed_motion_data = {}
    for action in data.domain.get_dict()["actions"]:
        #processed_motion_data[action["name"]]=[]
        motion_data[action["name"]] = {"0_observation": [], "1_command": []} 
    goal_encoded = encode_goal(data.problem)
    for action_data in data.motion_data_list:
        #output_task = encode_action(data.domain, action_data.action)
        action_encoded, location_encoded, object_encoded = encode_action(action_data.action)
        #print("Displacement_index", displacement_index)
        #print(action_data.action.name)
        #print(len(action_data.observation_list))
        #print(action_data.observation_list[-1])
        observation = action_data.observation_task
        input_task = np.concatenate([goal_encoded,observation])

        task_data["0_observation"].append(input_task)
        task_data["1_action"].append(action_encoded)
        task_data["2_locations"].append(location_encoded)
        task_data["3_objects"].append(object_encoded)

        input_motion = action_data.observation_motion
        command = action_data.command
        if input_motion is not None and command is not None:
            motion_data[action_data.action.name]["0_observation"].append(input_motion)
            motion_data[action_data.action.name]["1_command"].append(command)

    # save task dataset to hdf5
    save_to_hdf5(DATA_FOLDER+domain_name+"/"+env_name+"/task_data/"+file_name, task_data)

    # save motion dataset
    for action_name, action_value in motion_data.items():
        save_to_hdf5(DATA_FOLDER+domain_name+"/"+env_name+"/motion_data/"+action_name+"/"+file_name, action_value )
    # with safe_open(DATA_FOLDER+env_name+"/task_data/"+file_name,"wb") as f:
    #     #save in the environment
    #     pickle.dump(processed_task_data,f)
    # for key,value in processed_motion_data.items():
    #     with safe_open(DATA_FOLDER+env_name+"/motion_data/"+key+"/"+file_name,"wb") as f:
    #         pickle.dump(value,f)



if __name__ == "__main__":
    #a = Action(name="pick", parameters=["big_shelf","cup_red"])
    #a_encoded = encode_action(a)
    #print(a_encoded)
    
    #a_decoded = decode_action(a_encoded[0],a_encoded[1],a_encoded[2])
    #print(a_decoded)
    encode = encode_robot_holding(["plate_blue"])
    print(encode)
    action_list = {"0_observation": [], "1_command": []} 
    save_to_hdf5(DATA_FOLDER+"/abc.h5", action_list)