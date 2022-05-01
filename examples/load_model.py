from gil.policy.model.MotionNet import *
from gil.policy.model.TaskNet import *
from gil.data_processing.data_process import decode_action
import torch
import json

json_config = "/home/nkquynh/gil_ws/gil_hierachial_hmp/hierarchical-hmp/gil/data/configuration/policy_configuration.json"
with open(json_config,"r") as f:
    config = json.load(f)
#Create the net instance based on a name (string)
MODEL_FOLDER = config["saved_model_folder"]
param = config["tasknet"]["parameters"]
tasknet = eval(config["tasknet"]["type"])(param["goal_encoded_dim"], param["observation_dim"], param["action_dim"], param["object_dim"], param["location_dim"])
#Load network
tasknet.load_state_dict(torch.load(MODEL_FOLDER+config["tasknet"]["file_name"]))
tasknet.eval()
motion_net = {}

for action_name, action_net in config["motionnet"].items():
    param = action_net["parameters"]
    motion_net[action_name] = eval(action_net["type"])(param["observation_dim"],param["action_dim"])
    motion_net[action_name].load_state_dict(torch.load(MODEL_FOLDER+action_net["file_name"]))
    motion_net[action_name].eval()


# Test
a = torch.rand((1,38))
act,loc,obj = tasknet.forward(a)
print(act)