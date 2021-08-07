import numpy as np


class PongMath:
    def minimum_index(input, list_size):
        minimum_value = 10000
        for i in range(list_size):
            if minimum_value > input[i]:
                minimum_value = input[i]
                minimum_index = i
        return minimum_index

    # is used to return the highest probablity index
    def maximum_index(input, list_size):
        maximum_value = -100000
        for i in range(list_size):
            if maximum_value < input[i]:
                maximum_value = input[i]
                maximum_index = i
        return maximum_index

    def normalize(x_inputs):
        np_inputs = np.array(x_inputs)
        norm = np.linalg.norm(np_inputs)
        np_inputs = np_inputs / norm
        lowest_distance = min(np_inputs)
        np_inputs = np_inputs - lowest_distance
        return np_inputs


print("Output", PongMath.normalize([2, 3, 4, 2, 4]))
