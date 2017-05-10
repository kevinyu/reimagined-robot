from shapes.properties import properties
from shapes.objects import shapes
import numpy as np
import config


def label_at(scene_label, x, y, radius=config.LABEL_RADIUS):
    """Return one hot for object at position (x, y) """

    one_hot = np.zeros([len(shapes)] + [len(prop.params) for prop in properties])
    for obj_data in scene_label:
        obj_id, obj_x, obj_y, obj_props = obj_data  # obj_x, obj_y is the centers

        # ignore scale for now
        dist = np.power(x - obj_x, 2) + np.power(obj_y, 2)

        if dist < np.power(radius, 2):
            props = tuple([prop[1][0] for prop in obj_props])
            one_hot[(obj_id,) + props] = 1.0

    one_hot = one_hot.flatten()

    # final result should include one background element
    one_hot = np.concatenate([one_hot, np.array([0.0])])
    if one_hot.sum() == 0.0:
        one_hot[-1] = 1.0

    return one_hot / one_hot.sum()

'''
dat = [
        [0, 172.0, 323.0, [['Color', [1]], ['Scale', [2]], ['Rotation', [2]]]],
        [2, 365.0, 158.0, [['Color', [4]], ['Scale', [2]], ['Rotation', [1]]]],
        [3, 283.5, 461.5, [['Color', [6]], ['Scale', [0]], ['Rotation', [0]]]],
        [4, 155.0, 559.0, [['Color', [0]], ['Scale', [1]], ['Rotation', [0]]]],
        [0, 516.0, 419.0, [['Color', [0]], ['Scale', [1]], ['Rotation', [2]]]],
        [1, 646.5, 166.5, [['Color', [2]], ['Scale', [0]], ['Rotation', [2]]]],
        [2, 656.0, 383.0, [['Color', [2]], ['Scale', [2]], ['Rotation', [0]]]],
        [0, 517.5, 100.5, [['Color', [1]], ['Scale', [0]], ['Rotation', [0]]]],
        [3, 493.0, 254.0, [['Color', [5]], ['Scale', [1]], ['Rotation', [0]]]],
        [3, 78.5, 64.5, [['Color', [1]], ['Scale', [0]], ['Rotation', [1]]]],
        [4, 397.5, 726.5, [['Color', [4]], ['Scale', [0]], ['Rotation', [0]]]],
        [2, 640.5, 714.5, [['Color', [1]], ['Scale', [0]], ['Rotation', [1]]]],
        [3, 269.0, 646.0, [['Color', [6]], ['Scale', [1]], ['Rotation', [0]]]]
]

'''
