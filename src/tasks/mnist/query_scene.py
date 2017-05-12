import numpy as np

from properties import Color


u = np.pi/4.0
directions = {
    "below": 0.0,
    "below-right": 1.0 * u,
    "right": 2.0 * u,
    "above-right": 3.0 * u,
    "above": 4.0 * u,
    "above-left": -3.0 * u,
    "left": -2.0 * u,
    "below-left": -1.0 * u,
}


def distance(dx, dy):
    return np.sqrt(dx**2 + dy**2)


def angle(dx, dy):
    return np.arctan2(dy, dx)


def f(d, theta):
    return np.maximum(0.0, (np.abs(np.cos(theta)) > .6) * (50.0 / d) * np.cos(theta))


def query(scene_contents, row_idx, direction, threshold=0.2, speak=False):
    reference = scene_contents[row_idx]
    ref_x, ref_y = reference[1:3]

    if speak:
        print "whats", direction, "a", Color.params[reference[-1][0][1]], int(reference[0])

    weights = np.array([
        f(distance(ref_x - x, ref_y - y), angle(x - ref_x, y - ref_y) - directions[direction])
        for _, x, y, _ in scene_contents
    ])
    weights[row_idx] = 0

    result = weights / np.sum(weights)

    winner = np.argmax(result)
    if np.max(result) > threshold:
        if speak:
            print "its a", Color.params[scene_contents[winner][-1][0][1]], int(scene_contents[winner][0])
        return result
    else:
        if speak:
            print "nothing"
        return None

