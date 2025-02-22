MODEL_SCALES = {
    'Bowl': {'7000': 1, '7001': 1, '7002': 1, '7003': 1},
    'Cup': {'7004': 1, '7005': 1, '7006': 1, '7007': 1},
    'Mug': {'7008': 1, '7009': 1, '7010': 1, '7011': 1},
    'Pan': {'7012': 1, '7013': 1, '7014': 1, '7015': 1},
    'Fridge': {
        # '10144': 1.09,
        '10905': 1,
        '11299': 1,
        '11846': 1,
        '12036': 1,
        '12248': 1
    },
    'BottleTest': {
        # '3380': 0.2,
        '3517': 0.15,
        '3571': 0.17,
        '3763': 0.16,
        '3933': 0.16,
        '4043': 0.18,
        '4403': 0.1,
        '6771': 0.2,
        '8736': 0.15,
        '8848': 0.11
    },
    'Bottle': {
        '3558': 0.15,
        '3574': 0.16,
        '3614': 0.16,
        '3615': 0.18,
        '3616': 0.1,
        '3822': 0.2,
    },
    'Camera': {
        '101352': 0.08,
        '102411': 0.1,
        '102472': 0.1,
        '102434': 0.1,
        '102873': 0.1,
    },
    'EyeGlasses': {
        '101284': 0.15,
        '101287': 0.16,
        '101326': 0.16,
        '101293': 0.18,
        '101328': 0.16,
    },
    'Stapler': {
        '103100': 0.15,
        '103104': 0.15,
        # '103283': 0.13,
        '103299': 0.14,
        '103307': 0.08,
    },
    'MiniFridge': {
        '10797': 0.7,
        '10849': 0.7,
        '11178': 0.8,
        '11231': 0.8,
        '11709': 1,
        '12249': 0.7,
        '12252': 0.7
    },
    'Food': {
        'VeggieArtichoke': 0.014,
        'VeggieCabbage': 0.0043,  ## 0.0047 too big for rummy
        'VeggiePotato': 0.013,
        'VeggieSweetPotato': 0.013,  ## 0.015
        'VeggieTomato': 0.008,
        'VeggieZucchini': 0.01,

        ## able to render but not generate world with
        # 'MeatTurkeyLeg': 0.0007,
        'VeggieCauliflower': 0.0056,
        'VeggieGreenPepper': 0.0005,
    },
    'MeatTurkeyLeg': {
        '0001': 0.0007,
    },
    'Salter': {
        '3934': 0.05,
        '5861': 0.06,
        '6335': 0.06,
        '6493': 0.06,
        # 'BeehiveSalter': 0.02,
        # 'SquareSalter': 0.02,
        # 'TowerSalter': 0.02,
    },
    'Plate': {
        'SquarePlate': 0.025,
    },
    'Box': {
        '100129': 0.5,
        '100154': 0.5,
        '100162': 0.5,
        '100189': 0.5,
        '100194': 0.5,
        '100426': 0.5,
        '100234': 0.5,
        '100676': 0.5,
    },
    'Medicine': {
        '3520': 0.05,
        '3596': 0.1,
        '3655': 0.1,
        '4084': 0.05,
        '4427': 0.05,
        '4514': 0.05,
        '5861': 0.05,
        '6263': 0.05,
    },
    'Tray': {
        '100191': 0.5,
        '100685': 0.5,
        '100501': 0.3,
        '101176': 0.3,
    },
    'Knife': {
        '101052': 0.1,
        '101054': 0.1,
        '101057': 0.1,
        '101059': 0.1,
        '101085': 0.1,
        '101260': 0.1,
        '102400': 0.1,
    },
    'ShoppingCart': {
        '100491': 0.5,
        '100498': 0.5,
        '101066': 0.5,
        '102552': 0.5,
    },
    # 'Cart': {
    #     '100490': 0.5,
    #     '100496': 0.5,
    #     '100500': 0.5,
    #     '100501': 0.5,
    #     '100507': 0.5,
    #     '100509': 0.5,
    #     '101090': 0.5,
    #     '101083': 0.5,
    # },
    'Faucet': {
        '104': 0.3,
        '148': 0.15,
        '149': 0.3,
        '156': 0.2,
        '167': 0.2,
        '168': 0.2,
        '811': 0.2,
        '866': 0.2,
        '908': 0.3,
        '929': 0.15,
        '1280': 0.3,
    },
    'Kettle': {
        '101305': 0.3,
        '101311': 0.2,
        '101315': 0.2,
        '102714': 0.2,
        '102724': 0.2,
        '102726': 0.2,
    },
    'BraiserLid': {
        '100015': 0.259,
        '100017': 0.239,
        '100021': 0.229,
        '100023': 0.259,
        '100038': 0.259,
        '100045': 0.259,
        '100693': 0.259,
    },
    'BraiserBody': {
        '100015': 0.259,
        '100017': 0.239,
        '100021': 0.229,
        '100023': 0.259,
        '100038': 0.259,
        '100045': 0.259,
        '100693': 0.259,
    },
    'Oven': {
        '01909': 1,
        '101924': 1, '101940': 1, '101943': 1,
    },
    'OvenTop': {
        '102055': 1,
        '102060': 1,
    },
    'Toaster': {
        '103469': 0.2,
        '103473': 0.2,
        '103477': 0.2,
        '103482': 0.2,
        '103483': 0.2,
        '103485': 0.2,
        '103486': 0.2,
        '103553': 0.2,
        '103555': 0.2,
    },
    'CabinetTop': {
        '00001': 1,
        '00002': 1,
        '00003': 1
    },
    'TrashCan': {
        '10357': 0.2,
        '10584': 0.2,
        '11124': 0.2,
        '11361': 0.2,
        '11818': 0.2,
        '12445': 0.2,
        '101377': 0.2,
        '100732': 0.2,
    },
    'Sink': {
        '00004': 1.25,
        '00005': 0.005,
        '102379': 0.5,
        '1023790': 0.5,  ## without the base

        '100191': 0.35,
        '100685': 0.5,
        '100501': 0.45,
        '101176': 0.45,
    },
    'Basin': {
        '102379': 1,
    },
}
MODEL_SCALES['MiniFridgeDoorless'] = MODEL_SCALES['MiniFridge']
MODEL_SCALES.update({k.lower(): v for k, v in MODEL_SCALES.items()})
for c in ['Food', 'Plate', 'CabinetTop']:
    MODEL_SCALES[c].update({k.lower(): v for k, v in MODEL_SCALES[c].items()})

MODEL_HEIGHTS = {
    'KitchenCounter': {
        'height': 0.7,
        'models': ['32354', '33457', '32932', '32052', '30238', ## '32746', '23724',
                   '25959', '26608', '26387', '26800', '26657', '24152']
    },
    'CabinetUpper': {
        'height': 0.768,
        'models': ['45621', '45526', '46108', '46889', '48381', '49182', '46744']
    },
    'CabinetAboveOven': {
        'height': 0.4,
        'models': ['46408', '46427', '45633']
    },
    'CabinetLower': {
        'height': 1,
        'models': ['35059', '41004', '45213']
    },
    'SinkBase': {
        'height': 1,
        'models': ['44781', '45305', '45332', '45354', '45463', '46120', '46481'],
    },
    'CabinetTall': {
        'height': 2,
        'models': ['45594', '46197', '46236', '46456', '46896']
    },
    'DeskStorage': {
        'height': 0.6,
        'models': ['47235', '48517', '45001']
    },
    'DishwasherBox': {
        'height': 1,
        'models': ['11622', '11700', '12414', '12428', '12553']  ## '11826', 
    },
    'Dishwasher': {
        'height': 1,
        'models': ['2085', '12065', '12349', ]
    },
    'Microwave': {
      'height': 0.35,
      'models': ['7119', '7128', '7167', '7236', '7263', '7265', '7310']
    },
    'MicrowaveHanging': {
        'height': 0.6,
        'models': ['7292', '7296', '7349']  ## '7221', '7273', '7306', '7666'
    },
    'MiniFridgeBase': {
        'height': 0.6,
        'models': ['45166', '45261', '45526', '45746', '45850', '45963', '46166']
    },
    'OvenCounter': {
        'height': 1.05,
        'models': ['101773', '101808', '101917', '101921', '101930', '102044'],
    },
    'DiningTable': {
        'height': 1.05,
        'models': ['22301', '22339', '23372', '23807', '23782', '25308',
                   '25913', '28164', '32932', '34617'],
    },
    'DoorFrame': {
        'height': 2,
        'models': ['8867'],
    },
    'Cupboard': {
        'height': 2,
        'models': ['41085', '46490', '46563'],
    },
    'Bin': {
        'height': 0.5,
        'models': ['11361'],
    },
    'DinerTable': {
        'height': 0.7,
        'models': ['28668'],
    },
    'DinerChair': {
        'height': 1,
        'models': ['100568'],
    }
}
MODEL_HEIGHTS.update({k.lower(): v for k, v in MODEL_HEIGHTS.items()})

OBJ_SCALES = {
    'OilBottle': 0.25, 'VinegarBottle': 0.25, 'Salter': 0.1,
    'KitchenKnife': 0.1, 'KitchenFork': 0.2,
    'Pan': 0.3, 'Pot': 0.3, 'Bottle': 0.25,
    'Egg': 0.03, 'Veggie': 0.3, 'VeggieLeaf': 0.3, 'VeggieStem': 0.3,
    'MilkBottle': 0.2, 'Bucket': 0.7, 'Cart': 1.1, 'PotBody': 0.3,
    'PlateFat': 0.8, 'PlateFlat': 0.8, 'MeatTurkeyLeg': 0.0007
    # 'VeggieCabbage': 0.005, 'MeatTurkeyLeg': 0.0007, 'VeggieTomato': 0.005,
    # 'VeggieZucchini': 0.01, 'VeggiePotato': 0.015, 'VeggieCauliflower': 0.008,
    # 'VeggieGreenPepper': 0.0003, 'VeggieArtichoke': 0.017, 'MeatChicken': 0.0008,
}
OBJ_SCALES = {k.lower(): v * 0.7 for k, v in OBJ_SCALES.items()}

DONT_LOAD = [
    'VeggieCauliflower', 'VeggieGreenPepper', ## unable to load in gym
    'VeggieSweetPotato',  ## can't find grasp for feg
    '7265', ## microwave door partially open
    '46744', ## cabinet upper
    # '102379',  ## has an extra sink base
    # 'veggiezucchini', ## temporarily
]