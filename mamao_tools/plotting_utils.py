
color_dict = {
    'b': ('#3498db', '#2980b9'),
    'g': ('#2ecc71', '#27ae60'),
    'r': ('#e74c3c', '#c0392b'),
    'y': ('#f1c40f', '#f39c12'),
    'p': ('#9b59b6', '#8e44ad'),
    'gray': ('#95a5a6', '#7f8c8d'),
    'w': ('#ffffff',),
}
gradients = ['#4392ce', '#528bc1', '#6185b3', '#707fa6', '#7f7899', '#8e728b',
             '#9c6c7e', '#ab6571', '#ba5f64', '#c95957', '#d85249', '#e74c3c']
cc = ['b', 'r', 'y', 'gray']


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))