current_state = frozenset({('on', 'plate_blue', 'table'), ('on', 'plate_green', 'table'), ('on', 'cup_blue', 'table'), ('on', 'cup_green', 'table')})
goal_state =  {frozenset({('on', 'plate_green', 'table'), ('agent-free',), ('on', 'plate_blue', 'table'), ('on', 'cup_blue', 'table'), ('agent-at', 'table'), ('on', 'cup_green', 'table'), ('agent-avoid-human',)})}
current_state =  frozenset({('on', 'plate_green', 'table'), ('agent-free',), ('on', 'plate_blue', 'table'), ('on', 'cup_blue', 'table'), ('agent-at', 'table'), ('on', 'cup_green', 'table'), ('agent-avoid-human',)})
print (current_state in goal_state)
print(list(goal_state)[0])
