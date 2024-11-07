prompt_planning = """
Plan a short sequence of [OUTPUT] that accomplishes the following goal: 
``{goal}''. 
[RESPOND_WITH]
where <obj>, <surface>, <joint>, <button> and <handle> must be items from the following list:
{objects}.

Currently, you can see the following objects:
``{observed}''
{history}
You are a mobile robot with {n_arms}. You must obey the following commonsense rules:
1. You must have at least one empty hand before you can pick up an object or open or close a joint.
2. When you sprinkle or pour something into a container, there must not be objects placed on top of the container.
3. You can only take actions on objects that you can see. 
4. If you cannot see an object, it may be behind a door or inside drawer.
5. If you cannot see the inside of a space, you must open its door or drawer before you can pick objects from it or place objects inside it. 
"""

include_history = """
You have already taken the following actions written in a formal language:
{actions}

You just failed at planning for {failure}.

"""

## ---------------------------------------- PREDICT SUBGOALS ---------------------------------------------------

prompt_subgoals_english = (prompt_planning.replace('[OUTPUT]', 'intermediate goals')
                           .replace('<obj>, <surface>, <joint>, <button> and <handle>', 'objects mentioned')
                           .replace('[RESPOND_WITH]', """
Respond with detailed but simple instructions in English. Each line must consists of only one intermediate goal, """))

## 'under(<obj>, <obj>)': the result of picking up <obj> and placing it under <obj>,
## 'poured-to(<movable>, <region>)': the result of picking up <movable> and pouring it into <region>, it contains two arguments.
prompt_english_to_subgoals = """
Translate the above intermediate goals into a formal language defined by the following subgoals.

subgoals = 
[
'picked(<movable>)': the result of picking up <movable>, it contains one argument.
'in(<movable>, <space>)': the result of picking up <movable> and placing it inside <space>, it contains two arguments.
'on(<movable>, <surface>)': the result of picking up <movable> and placing it on <surface>, it contains two arguments.
'sprinkled-to(<movable>, <region>)': the result of picking up <movable> and sprinkling it into <region>, it contains two arguments.
'stirred(<region>, <movable>)': the result of picking up <movable> to stir <region>, it contains two arguments.
'chopped(<movable>, <utensil>)': the result of picking up <utensil> to chop <movable>, it contains two arguments.
'opened-door(<door>)': the result of opening <door>, it contains one argument.
'closed-door(<door>)': the result of closing <door>, it contains one argument.
'opened-drawer(<drawer>)': the result of closing <drawer>, it contains one argument.
'closed-drawer(<drawer>)': the result of closing <drawer>, it contains one argument.
'pressed(<button>)': the result of pressing <button>, it contains one argument.
'turned-on(<knob>)': the result of turning <handle> or <knob> to start up the associated appliance, it contains one argument.
'turned-off(<knob>)': the result of turning <handle> or <knob> to shut down the associated appliance, it contains one argument.
], 

The above subgoals include argument types. Please use the objects in the respective types:
``
{objects}
''

Return the subgoals in a list and give no explanation. 
Make sure the sub-goals are in the same order as the steps in the intermediate goals. 
Note that the arguments shouldn't include robot parts, e.g., 'arm', 'gripper'.
If a new object not mentioned in the set of objects is used as arguments, please name it with the same as the object that constitutes it the most and exists in the given list of objects.
If one intermediate goal cannot be translated into a sub-goal, skip that step.
"""

## ---------------------------------------- PREDICT ACTIONS ---------------------------------------------------

prompt_actions_english = (prompt_planning.replace('[OUTPUT]', 'actions')
                          .replace('<obj>, <surface>, <joint>, <button> and <handle>', 'objects mentioned')
                          .replace('[RESPOND_WITH]', """
Respond with detailed but simple instructions in English. Each line must consists of only one action, """))

## 'under(<obj>, <obj>)': the result of picking up <obj> and placing it under <obj>,
prompt_english_to_actions = """
Translate the each of the listed actions in English into a formal language defined by the following primitive actions. 
Each action in English may correspond to multiple primitive actions:
{set_of_actions}

The above actions include argument types. Please use the objects in the respective types:
``
{objects}
''

Return the actions in a list and give no explanation. 
Note that:
1. The arguments shouldn't include robot parts, e.g., 'arm', 'gripper'.
2. If a new object not mentioned in the set of objects is used as arguments, please name it with the same as the object that constitutes it the most and exists in the given list of objects.
3. If one action cannot be translated into the above set, skip that action.
"""
## 'pour(<obj>, <container>)': it contains two arguments.
list_of_actions_with_preconditions = """
actions = 
[
'pick(<obj>)': it contains one argument. The robot must have an empty hand to pick up an object.
'place(<obj>, <surface>)': it contains two arguments.
'sprinkle(<obj>, <container>)': it contains two arguments. There must be no object placed on top of the container.
'stir(<container>, <obj>)': it contains two arguments. There must be no object placed on top of the container.
'chop(<obj>, <utensil>)': it contains two arguments.
'open(<joint>)': it contains one argument.
'close(<joint>)': it contains one argument.
'nudge(<door>)': it contains one argument. This action is applied to open a door even wider.
'turn-on(<knob>)': turning on a stove knob, it contains one argument.
'turn-off(<knob>)': turning off a stove knob, it contains one argument.
'turn-on(<handle>)': turning on a faucet handle, it contains one argument.
'turn-off(<handle>)': turning off a faucet handle, it contains one argument.
], 
"""

list_of_actions = """
actions = 
[
'pick(<obj>)': it contains one argument. 
'place(<obj>, <surface>)': it contains two arguments.
'pour(<obj>, <container>)': it contains two arguments.
'sprinkle(<obj>, <container>)': it contains two arguments.
'stir(<container>, <obj>)': it contains two arguments.
'chop(<obj>, <utensil>)': it contains two arguments.
'open(<joint>)': it contains one argument.
'close(<joint>)': it contains one argument.
'nudge(<door>)': it contains one argument. 
'turn-on(<knob>)': it contains one argument.
'turn-off(<knob>)': it contains one argument.
], 
"""

## ---------------------------------------- IMAGE DESCRIPTION ---------------------------------------------------

default_image_description = """
The accompanying image shows a scene with a robot in a kitchen.
"""

composed_annotated_image_description = """
The accompanying image is a collage of two images depicting a scene with a robot in a kitchen. 
There are different sets of annotations of object names with object bounding boxes drawn on the images.
"""


## --------------------------- PREDICT CONTINUOUS STATES (not used in final version) ----------------------------------

prompt_english_to_object_states = """
Translate each line into a sequence of changes in object states, i.e., object poses and joint positions, during the process of achieving the intermediate goal.

Note that 3D poses are represented as `<obj> is at (x, y, z)`. Joint positions are represented as `<joint> is at angle` where angle is in radians.

Current object state: 
{current_state}

Note that the following objects won't change poses: 
{static_object_state}

Note that joint positions must be within the limits as follows: 
{joint_limits}

Return the list of translated state changes and give no explanation. Each line is a sequences of state changes. Ignore the poses of objects and positions of joints that doesn't change during the process of achieving the intermediate goal.
Make sure the lines of state changes are in the same order as the steps in the intermediate goals. 
If no state change happens in a line, write `no change` for that line."""

prompt_translate_subgoals_to_english = """
Translate the subgoals to instructions in English. Return a list."""

prompt_goal_achieved = """
Determine whether the goal ``{goal}'' is achieved. 
Return 0 or 1, where 0 means the goal will not be achieved by following the instructions, while 1 means the goal could be achieved."""

prompt_goal_repair = """
Revise the English instructions so that the goal could be achieved. 
Return a list of English instructions, along with explanations why some steps are revised."""