from world_builder.loaders import *
from nsplan_tools.nsplan_utils import debug_print


def sample_clean_dish_v0(world, w=3, l=8, verbose=True, pause=True):

    # DEBUG:
    # TODO: instead of making it one scale, sample different scales for sink base, sink, and faucet. but make sure they
    #       are compatible
    random_scale = np.random.uniform(0.7, 1.3)
    random_scale = 1.0
    print("random scale:", random_scale)

    h_lower_cabinets = 1 * random_scale
    dh_cabinets = 0.8 * random_scale
    h_upper_cabinets = 0.768 * random_scale

    global COUNTER_THICKNESS
    COUNTER_THICKNESS = COUNTER_THICKNESS * random_scale

    wall_height = h_lower_cabinets + dh_cabinets + h_upper_cabinets + COUNTER_THICKNESS

    floor = create_house_floor(world, w=w, l=l, x=w/2, y=l/2)

    ordering = sample_kitchen_furniture_ordering()
    while 'SinkBase' not in ordering:
        ordering = sample_kitchen_furniture_ordering()
    # debug: use this fixed ordering for now
    ordering = ['CabinetLower', 'SinkBase', 'DishwasherBox']

    """ step 1: sample a sink """
    start = ordering.index('SinkBase')
    sink_y = l * start / len(ordering) + np.random.normal(0, 0.5)
    floor, base, counter_x, counter_w, counter_z, color, counters = \
        sample_kitchen_sink(world, floor=floor, y=sink_y, random_scale=random_scale)

    under_counter = ['SinkBase', 'CabinetLower', 'DishwasherBox']
    on_base = ['MicrowaveHanging', 'MiniFridge']
    full_body = ['CabinetTall', 'Fridge', 'OvenCounter']
    tall_body = ['CabinetTall', 'Fridge', 'MiniFridge']

    def update_x_lower(obj, x_lower):
        if obj.aabb().lower[0] < x_lower:
            x_lower = obj.aabb().lower[0]
        return x_lower

    def load_furniture(category, random_scale=1.0):
        ins = True
        if world.note in [551, 552]:
            if category == 'MiniFridge':
                ins = random.choice(['11178', '11231'])  ## two doors
            if category == 'CabinetTop':
                ins = random.choice(['00003'])  ## two doors
            if category == 'Sink':
                ins = random.choice(['00003'])  ## two doors
        if world.note in [553]:
            if category == 'OvenCounter':
                ins = random.choice(['101921'])  ## two doors
        if world.note in [555]:
            if category == 'MiniFridge':
                ins = random.choice(['11709'])  ## two doors
        return world.add_object(Object(
            load_asset(category, yaw=math.pi, floor=floor, random_instance=ins, verbose=True, random_scale=random_scale),
            name=category, category=category))

    def load_furniture_base(furniture):
        return world.add_object(Object(
            load_asset('MiniFridgeBase', l=furniture.ly, yaw=math.pi, floor=floor,
                       random_instance=True, verbose=True),
            name=f'{furniture.category}Base', category=f'{furniture.category}Base'))

    counter_regions = []
    tall_obstacles = []
    right_counter_lower = right_counter_upper = base.aabb().upper[1]
    left_counter_lower = left_counter_upper = base.aabb().lower[1]
    x_lower = base.aabb().lower[0]

    adjust_y = {}

    """ step 2: on the left and right of sink base, along with the extended counter """

    # TODO: also apply the random scale that is applied to sink
    for direction in ['+y', '-y']:
        if direction == '+y':
            categories = [c for c in ordering[start+1:]]
        else:
            categories = [c for c in ordering[:start]][::-1]
        current = base
        for category in categories:
            adjust = {}  ## doors bump into neighbors and counters
            furniture = load_furniture(category, random_scale=random_scale)
            if category in tall_body:
                tall_obstacles.append(furniture)

            if category in full_body + on_base:
                if direction == '+y' and right_counter_lower != right_counter_upper:
                    counter_regions.append([right_counter_lower, right_counter_upper])
                elif direction == '-y' and left_counter_lower != left_counter_upper:
                    counter_regions.append([left_counter_lower, left_counter_upper])

            ## x_lower aligns with the counter and under the counter
            if category in under_counter + full_body:
                adjust = furniture.adjust_next_to(current, direction=direction, align='+x')

            ## put a cabinetlower with the same y_extent as the object
            elif category in on_base:
                if furniture.mobility_id not in ['11709']:
                    furniture_base = load_furniture_base(furniture)
                    adjust = furniture_base.adjust_next_to(current, direction=direction, align='+x')
                    furniture.adjust_next_to(furniture_base, direction='+z', align='+x')
                    x_lower = update_x_lower(furniture_base, x_lower)
                else:
                    adjust = furniture.adjust_next_to(current, direction=direction, align='+x')
                    # counters.append(furniture)
                    world.add_to_cat(furniture.body, 'supporter')
                x_lower = update_x_lower(furniture, x_lower)

            adjust_y.update(adjust)

            if direction == '+y':
                right_counter_upper = furniture.aabb().upper[1]
            else:
                left_counter_lower = furniture.aabb().lower[1]

            x_lower = update_x_lower(furniture, x_lower)
            current = furniture
            if category in full_body + on_base:
                if direction == '+y':
                    right_counter_lower = right_counter_upper
                else:
                    left_counter_upper = left_counter_lower
            # if direction == '+y':
            #     right_most = furniture.aabb().upper[1]
            # else:
            #     left_most = furniture.aabb().lower[1]
    if right_counter_lower != right_counter_upper:
        counter_regions.append([right_counter_lower, right_counter_upper])
    if left_counter_lower != left_counter_upper:
        counter_regions.append([left_counter_lower, left_counter_upper])

    ## adjust counter regions
    new_counter_regions = []
    for lower, upper in counter_regions:
        original = [lower, upper]
        if lower in adjust_y:
            lower = adjust_y[lower]
        if upper in adjust_y:
            upper = adjust_y[upper]
        new_counter_regions.append([lower, upper])
    counter_regions = new_counter_regions

    # Debug: remove minifridge for now
    ## make doors easier to open
    # world.name_to_object('minifridge').adjust_pose(dx=0.2, world=world)

    ## make wall
    l = right_counter_upper - left_counter_lower
    y = (right_counter_upper + left_counter_lower) / 2
    x = x_lower - WALL_WIDTH / 2
    wall = world.add_object(
        Supporter(create_box(w=WALL_WIDTH, l=l, h=wall_height, color=color), name='wall'),
        Pose(point=Point(x=x, y=y, z=wall_height/2)))
    floor.adjust_pose(dx=x_lower - WALL_WIDTH, world=world)

    """ step 3: make all the counters """
    sink_left = world.name_to_object('sink_counter_left')
    sink_right = world.name_to_object('sink_counter_right')

    def could_connect(y1, y2, adjust_y):
        if equal(y1, y2):
            return True
        result1 = in_list(y1, adjust_y)
        if result1 is not None:
            if equal(adjust_y[result1], y2):
                return True
        result2 = in_list(y2, adjust_y)
        if result2 is not None:
            if equal(adjust_y[result2], y1):
                return True
        return False

    for lower, upper in counter_regions:
        name = 'counter'
        if could_connect(lower, sink_right.aabb().upper[1], adjust_y):
            name = 'sink_counter_right'
            lower = sink_right.aabb().lower[1]
            counters.remove(sink_right)
            world.remove_object(sink_right)
        elif could_connect(upper, sink_left.aabb().lower[1], adjust_y):
            name = 'sink_counter_left'
            upper = sink_left.aabb().upper[1]
            counters.remove(sink_left)
            world.remove_object(sink_left)
        counters.append(world.add_object(
            Supporter(create_box(w=counter_w, l=upper-lower,
                                 h=COUNTER_THICKNESS, color=color), name=name),
            Pose(point=Point(x=counter_x, y=(upper + lower) / 2, z=counter_z))))
        # print('lower, upper', (round(lower, 2), round(upper, 2)))

    ## to cover up the wide objects at the back
    if x_lower < base.aabb().lower[0]:
        x_upper = base.aabb().lower[0]
        x = (x_upper+x_lower)/2
        counter_regions.append([base.aabb().lower[1], base.aabb().upper[1]])

        ## merge those could be merged
        counter_regions = sorted(counter_regions, key=lambda x: x[0])
        merged_counter_regions = [counter_regions[0]]
        for i in range(1, len(counter_regions)):
            if could_connect(counter_regions[i][0], merged_counter_regions[-1][1], adjust_y):
                merged_counter_regions[-1][1] = counter_regions[i][1]
            else:
                merged_counter_regions.append(counter_regions[i])

        for lower, upper in merged_counter_regions:
            world.add_object(
                Object(create_box(w=x_upper - x_lower, l=upper - lower,
                                  h=COUNTER_THICKNESS, color=color),
                       name='counter_back', category='filler'),
                Pose(point=Point(x=x, y=(upper + lower) / 2, z=counter_z)))
            # print('lower, upper', (round(lower, 2), round(upper, 2)))

    # Debug: not adding this for now
    """ step 4: put upper cabinets and shelves """
    # oven = world.name_to_object('OvenCounter')
    # left_counter_lower and right_counter_upper are y values

    # TODO: apply random scale
    cabinets, shelves = load_full_kitchen_upper_cabinets(world, counters, x_lower, left_counter_lower,
                                                         right_counter_upper, others=[], #DEBUG=[oven],
                                                         dz=dh_cabinets, obstacles=tall_obstacles,
                                                         random_scale=random_scale)

    """ step 5: add additional surfaces in furniture """
    sink = world.name_to_object('sink')
    sink_bottom = world.add_surface_by_keyword(sink, 'sink_bottom')

    # weiyu debug
    # sink_left = world.name_to_object('sink_counter_left')
    # sink_right = world.name_to_object('sink_counter_right')
    # counter_left = world.add_surface_by_keyword(sink_left, 'counter_left')
    # counter_right = world.add_surface_by_keyword(sink_right, 'counter_right')

    # # Debug: not adding this for now
    # """ step 5: place electronics and cooking appliances on counters """
    only_counters = [c for c in counters]
    obstacles = []
    # microwave = None
    # if 'MicrowaveHanging' not in ordering:
    #     wide_counters = [c for c in counters if c.ly > 0.66]
    #     if len(wide_counters) > 0:
    #         counter = wide_counters[0]
    #         microwave = counter.place_new_obj('microwave', scale=0.4 + 0.1 * random.random(),
    #                                           random_instance=True, verbose=True, world=world)
    #         microwave.set_pose(Pose(point=microwave.get_pose()[0], euler=Euler(yaw=math.pi)), world=world)
    #         obstacles.append(microwave)
    # else:
    #     microwave = world.name_to_object('MicrowaveHanging')
    # # if microwave is not None:
    # #     counters.append(microwave)
    # #     world.add_to_cat(microwave.body, 'supporter')

    x_food_min = base.aabb().upper[0] - 0.3
    # Debug: not adding this for now
    # braiser, braiser_bottom = load_braiser(world, oven, x_min=x_food_min)
    # obstacles.extend([braiser, braiser_bottom])

    """ step 5: place movables on counters """
    # all_counters = {
    #     'food': counters,
    #     'bottle': counters + [sink_bottom],
    #     'medicine': shelves + [microwave],
    # }
    # Debug: no microwave for now
    all_counters = {
        'food': counters,
        'bottle': counters + [sink_bottom],
        'medicine': shelves,
        'bowl': counters,
        'mug': counters,
        'pan': counters,
    }
    possible = []
    for v in all_counters.values():
        possible.extend(v)

    ## draw boundary of surfaces
    drawn = []
    for c in possible:
        if c in drawn: continue
        mx, my, z = c.aabb().upper
        aabb = AABB(lower=(x_food_min, c.aabb().lower[1], z), upper=(mx, my, z + 0.1))
        draw_aabb(aabb)
        drawn.append(str(c))

    ## probility of each door being open
    world.make_doors_transparent()
    # Debug: open all doors
    epsilon = 1.0
    # Debug: no fridge for now
    # load_storage_mechanism(world, world.name_to_object('minifridge'), epsilon=epsilon)
    for cabi_type in ['cabinettop', 'cabinetupper']:
        cabi = world.cat_to_objects(cabi_type)
        if len(cabi) > 0:
            cabi = world.name_to_object(cabi_type)

            # TODO: instead of adding space, add surface in cabinet
            load_storage_mechanism(world, cabi, epsilon=epsilon)

    # add space for dishwasher
    # load_storage_mechanism(world, world.name_to_object('dishwasherbox'), epsilon=epsilon)

    ## load objects into reachable places
    food_ids, bottle_ids, medicine_ids, bowl_ids, mug_ids, pan_ids = \
        load_counter_movables(world, all_counters, d_x_min=0.3, obstacles=obstacles)
    movables = food_ids + bottle_ids + medicine_ids + bowl_ids + mug_ids + pan_ids

    """ step 6: take an image """
    set_camera_pose((4, 4, 3), (0, 4, 0))

    # debug_print(sample_clean_dish_v0, "built")
    # wait_unlocked()

    # pause = True
    if pause:
        wait_unlocked()
    return movables, cabinets, only_counters, obstacles, x_food_min