MODEL_SCALES = {
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
        '3763': 0.16,
        '3933': 0.16,
        '4043': 0.18,
        '4403': 0.1,
        '6771': 0.2,
        '8736': 0.15,
        '8848': 0.11
    },
    'Bottle': {
        # '3380': 0.2,
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
    'Glasses': {
        '101284': 0.15,
        '101287': 0.16,
        '101326': 0.16,
        '101293': 0.18,
        '101328': 0.1,
    },
    'Stapler': {
        '103100': 0.15,
        '103104': 0.16,
        '103283': 0.16,
        '103299': 0.14,
        '103307': 0.08,
    },
    'MiniFridge': {
        '10797': 1,
        '10849': 1,
        '11178': 1,
        '11231': 1,
        '11709': 1,
        '12249': 1,
        '12252': 1
    },
    'Food': {
        'VeggieCabbage': 0.0047,
        'MeatTurkeyLeg': 0.0007,
        'VeggieZucchini': 0.01,
        'VeggieSweetPotato': 0.015,
        'VeggiePotato': 0.013,
        'VeggieCauliflower': 0.008,
        'VeggieArtichoke': 0.014,
        'VeggieGreenPepper': 0.0005,
        'VeggieTomato': 0.008,
    },
    'Salter': {
        '3934': 0.05,
        '5861': 0.06,
        '6335': 0.06,
        '6493': 0.06,
    },
    'Plate': {
        'SquarePlate': 0.02,
    }
}
MODEL_SCALES['MiniFridgeDoorless'] = MODEL_SCALES['MiniFridge']
MODEL_SCALES.update({k.lower(): v for k, v in MODEL_SCALES.items()})
MODEL_SCALES['Food'].update({k.lower(): v for k, v in MODEL_SCALES['Food'].items()})
MODEL_SCALES['Plate'].update({k.lower(): v for k, v in MODEL_SCALES['Plate'].items()})

MODEL_HEIGHTS = {
    'KitchenCounter': {
        'height': 0.7,
        'models': ['32354', '33457', '32932', '32052', '30238', ## '32746', '23724',
                   '25959', '26608', '26387', '26800', '26657', '24152']
    },
    'KitchenMicrowave': {
        'height': 0.6,
        'models': ['7310', '7320', '7221', '7167', '7263', '7119']
    }
}
MODEL_HEIGHTS.update({k.lower(): v for k, v in MODEL_HEIGHTS.items()})

OBJ_SCALES = {
    'OilBottle': 0.25, 'VinegarBottle': 0.25, 'Salter': 0.1, 'Knife': 0.1, 'Fork': 0.2,
    'Microwave': 0.7, 'Pan': 0.3, 'Pot': 0.3, 'Kettle': 0.3, 'Bottle': 0.25,
    'Egg': 0.03, 'Veggie': 0.3, 'VeggieLeaf': 0.3, 'VeggieStem': 0.3,
    'MilkBottle': 0.2, 'Toaster': 0.2, 'Bucket': 0.7, 'Cart': 1.1,
    'PotBody': 0.3, 'BraiserBody': 0.37, 'BraiserLid': 0.37, 'Faucet': 0.35,
    'VeggieCabbage': 0.005, 'MeatTurkeyLeg': 0.0007, 'VeggieTomato': 0.005,
    'VeggieZucchini': 0.01, 'VeggiePotato': 0.015, 'VeggieCauliflower': 0.008,
    'VeggieGreenPepper': 0.0003, 'VeggieArtichoke': 0.017, 'MeatChicken': 0.0008,
    'PlateFat': 0.8, 'PlateFlat': 0.8
}
OBJ_SCALES = {k.lower(): v * 0.7 for k, v in OBJ_SCALES.items()}