import os
import sys
from os.path import join, abspath, dirname, isdir, isfile, pardir
ROOT_DIR = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([
    join(ROOT_DIR),
])

import json

from vlm_tools.vlm_api import GPT4vApi
from vlm_tools.vlm_utils import EXP_DIR


def test_gpt4v():
    """
----------------------------------------
how to make a sandwich? tell me one recipe. answer with a list of steps.
----------------------------------------
["Sure! Here's a simple recipe for a classic turkey and cheese sandwich:\n\n### Ingredients:\n- 2 slices of bread (your choice: white, whole wheat, sourdough, etc.)\n- 2-3 slices of turkey breast\n- 1-2 slices of cheese (cheddar, Swiss, or your favorite type)\n- Lettuce leaves\n- Tomato slices\n- Mayonnaise\n- Mustard (optional)\n- Salt and pepper (optional)\n\n### Steps:\n1. **Prepare the Ingredients:**\n   - Wash and dry the lettuce leaves.\n   - Slice the tomato into thin rounds.\n\n2. **Toast the Bread (Optional):**\n   - If you prefer a toasted sandwich, lightly toast the bread slices in a toaster or on a skillet until they are golden brown.\n\n3. **Spread the Condiments:**\n   - Lay the bread slices on a clean surface.\n   - Spread a thin layer of mayonnaise on one or both slices of bread.\n   - If you like mustard, spread a thin layer on one of the slices as well.\n\n4. **Assemble the Sandwich:**\n   - Place the turkey slices evenly on one slice of bread.\n   - Add the cheese slices on top of the turkey.\n   - Layer the lettuce leaves over the cheese.\n   - Add the tomato slices on top of the lettuce.\n   - Sprinkle a little salt and pepper over the tomatoes if desired.\n\n5. **Close the Sandwich:**\n   - Place the second slice of bread on top of the assembled ingredients, condiment side down.\n\n6. **Cut and Serve:**\n   - If you prefer, cut the sandwich in half diagonally or into quarters.\n   - Serve immediately and enjoy!\n\nFeel free to customize this recipe with your favorite ingredients and condiments. Enjoy your sandwich!"]
... in 5.988 sec (gpt-4o) / 9.193 sec (gpt-4v)


----------------------------------------
Where is vinegar bottle?

Your answer should be a PLACE,
which includes a preposition from [in, on, above, below, next to]
and objects from ['fridge', 'cabinet'].

If you cannot see the object, answer with three PLACEs that it could be in. You should answer the more likely PLACEs first.
----------------------------------------
['The vinegar bottle could be:\n\n1. In the cabinet\n2. In the fridge\n3. Next to the fridge']
... in 1.219 sec (gpt-4o) / 1.944 sec (gpt-4v)
    """
    agent = GPT4vApi()
    agent.new_session()
    agent.ask('how to make a sandwich? tell me one recipe. answer with a list of steps.')
    # agent.ask('how to make this sandwich? tell me the steps in detail.', image_name='sandwich.jpg')
    # agent.ask('what is the best way to make a sandwich?', image_name='sandwich.jpg')

    # agent.ask_by_template('where', dict(object='vinegar bottle', objects=['fridge', 'cabinet']))


def test_gpt4v_chicken_soup():
    """
----------------------------------------
Plan a short sequence of intermediate goals that accomplishes the following goal:
``make chicken soup''.

Respond with detailed but simple instructions in English. Each line must consists of only one action,
where objects mentioned must be one item from the following list:
['side cabinet right door', 'side cabinet left door', 'top drawer', 'side cabinet', 'stove', 'counter top on the right', 'top drawer space', 'pot body', 'pot bottom', 'pot lid', 'stove knob', 'fridge', 'fridge door', 'chicken leg', 'salt shaker', 'pepper shaker', 'fork'].

Currently, you can see the following objects:
``the pot body is on the stove,
the pot lid is on the pot body,
the chicken leg is on the fridge's shelf top,
the salt shaker is on the side cabinet,
the pepper shaker is on the side cabinet,
the fork is in the top drawer space''

You must obey the following commonsense rules:
1. After you pick up an object, you need to place it somewhere.
2. You can only take actions on objects that you can see.
3. If you cannot see an object, it may be inside a door or drawer.
4. If you cannot see the inside of a space, you must open its door or drawer before you can pick objects from it or place objects inside it.

----------------------------------------
["1. Open the fridge door.\n2. Pick up the chicken leg from the fridge's shelf top.\n3. Close the fridge door.\n4. Place the chicken leg on the counter top on the right.\n5. Open the side cabinet right door.\n6. Pick up a pot bottom from the side cabinet.\n7. Close the side cabinet right door.\n8. Place the pot bottom on the stove.\n9. Pick up the pot body from the stove.\n10. Place the pot body on the pot bottom.\n11. Pick up the chicken leg from the counter top on the right.\n12. Place the chicken leg in the pot body.\n13. Pick up the salt shaker from the side cabinet.\n14. Shake salt into the pot body.\n15. Place the salt shaker back on the side cabinet.\n16. Pick up the pepper shaker from the side cabinet.\n17. Shake pepper into the pot body.\n18. Place the pepper shaker back on the side cabinet.\n19. Pick up the pot lid from the pot body.\n20. Place the pot lid on the pot body.\n21. Turn the stove knob to turn on the stove.\n22. Wait for the soup to cook.\n23. Turn the stove knob to turn off the stove.\n24. Pick up the fork from the top drawer space.\n25. Use the fork to check if the chicken leg is cooked.\n26. Place the fork back in the top drawer space."]
... in 3.927 sec (gpt-4o) / 15.623 sec (gpt-4v)

    """
    agent = GPT4vApi()
    agent.new_session()
    # agent.ask_by_template('how to make a sandwich? tell me one recipe. answer with a list of steps.')
    # agent.ask('how to make this sandwich? tell me the steps in detail.', image_name='sandwich.jpg')
    # agent.ask('what is the best way to make a sandwich?', image_name='sandwich.jpg')
    # agent.ask_by_template('where', dict(object='vinegar bottle', objects=['fridge', 'cabinet']))

    exp_dir = join(EXP_DIR, 'test_llm_kitchen_chicken_soup')
    exp_name = '240512_061712_default'
    memory_file = join(exp_dir, exp_name, 'llm_memory.json')
    memory = json.load(open(memory_file, 'r'))
    question = memory['subgoals_english']['prompt']

    agent.ask(question)
    print()
    print(agent.query_time)


if __name__ == '__main__':
    # test_gpt4v()
    test_gpt4v_chicken_soup()
