import json

f = open('track_history2.json', 'r')

#i need to write a prog which takes each objject ID, which is a value like 432: append it with 631

new_data = {}

for obj_id, traj_list in json.load(f).items():
    new_key = int(obj_id) + 937
    new_data[str(new_key)] = traj_list

with open('track_history3.json', 'w') as f:
    json.dump(new_data, f, indent=4)