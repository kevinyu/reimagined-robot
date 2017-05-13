import numpy as np
import theano
import theano.tensor as T

from properties import Color
from parameters import D_table
from utils import float_x, init_hypervectors
from utils.complex import ComplexTuple


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

D_directions = {}
learn_directions = []
for key, val in directions.items():
    D_directions[key] = ComplexTuple(
        init_hypervectors(1),
        init_hypervectors(1)
    )
    learn_directions.append(D_directions[key].real)
    learn_directions.append(D_directions[key].imag)

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


from parameters import _D_combined
D_real, D_imag = _D_combined.real, _D_combined.imag


def generate_queries(scene, n):
    """Takes a scene and generates n queries with distributions to match in the response
    
    Takes in scene, n
    
    Returns
    query (list): a series of operations to perform
    digit_result (np.ndarray, n): softmax over digits
    prop_results (np.ndarray, n x n_properties): softmax over each property
    """
    queries = []
    digit_result = []
    prop_results = []

    digit_labels = []
    color_labels = []

    for _ in range(n):
        idx = np.random.choice(range(len(scene.contents)))
        direction = np.random.choice(directions.keys())
        result = query(scene.contents, idx, direction, speak=False)

        digit_label = np.zeros(11)
        color_label = np.zeros(5)

        if result is not None:
            for i, weight in enumerate(result[result > 0]):
                digit_id, _, _, props = scene.contents[i]
                digit_label[int(digit_id)] += weight
                color_label[props[0][1]] += weight
        else:
            digit_label[-1] = 1
            color_label[-1] = 1

        # what is to the [direction] of [scene.contents[idx]]???
        ref_digit, _, _, ((_, ref_color),) = scene.contents[idx]

        ref_digit = int(ref_digit)
        ref_color = int(ref_color)
        # the first thing is what you bidn to S S*
        # the second is what you dot with

        queries.append(
                D_directions[direction].conj *
                ComplexTuple(
                    D_real[ref_digit, ref_color],
                    D_imag[ref_digit, ref_color]
                )
        )
        digit_labels.append(digit_label / np.sum(digit_label))
        color_labels.append(color_label / np.sum(color_label))

    queries_real = T.stack([q.real for q in queries], axis=2)
    queries_imag = T.stack([q.imag for q in queries], axis=2)
    queries = ComplexTuple(queries_real, queries_imag)

    digit_labels = np.stack(digit_labels, axis=1)
    color_labels = np.stack(color_labels, axis=1)

    return queries, (D_table["Digits"], digit_labels), (D_table["Color"], color_labels)

# N x n_queries : matrix to multiply with scene memory
# N x n_queries * N
# table: N x n_choices : matrix to dot agains
# n_queries x N dot N x n_choices
# n_queries x n_choices
# cross entropy with labels (n_queries x n_choices)


