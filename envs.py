import numpy as np

class Env_map():

    def __init__(self, adversary_map):
        self.initial_map = adversary_map

    def one_hot_map(self, char_map):
        new_map = np.zeros(self.initial_map.shape, dtype=np.float32)
        new_map[0][char_map == 'P'] = 1
        new_map[1][char_map == 'H'] = 1
        new_map[2][char_map == 'G'] = 1
        for i in range(self.initial_map.shape[0]):
            new_map[i][char_map == 'F'] = 0
        return new_map

    def deone_hot_map(self, one_hotted_map):
        new_map = np.zeros((self.initial_map.shape[1],self.initial_map.shape[2]), dtype='U1')
        new_map[one_hotted_map[0] == 1] = 'P'
        new_map[one_hotted_map[1] == 1] = 'H'
        new_map[one_hotted_map[2] == 1] = 'G'
        new_map[new_map == ''] = 'F'
        return new_map

    def deone_hot_map_with_start(self, one_hotted_map):
        new_map = np.zeros((self.initial_map.shape[1],self.initial_map.shape[2]), dtype='U1')
        new_map[one_hotted_map[0] == 1] = 'S'
        new_map[one_hotted_map[1] == 1] = 'H'
        new_map[one_hotted_map[2] == 1] = 'G'
        new_map[new_map == ''] = 'F'
        return new_map

    def map_step(self, observation):
        state_map = self.initial_map.copy()
        # remove start position
        state_map[0][state_map[0] == 1] = 0
        # place in current position
        state_map[0][calculate_coordinates(observation, self.initial_map.shape[1])] = 1
        char_map = self.deone_hot_map(state_map)
        return char_map, state_map

    def vectorized_map_step(self, observations):
        char_maps, state_maps = [], []
        for observation in observations:
            state_map = self.initial_map.copy()
            # remove start position
            state_map[0][state_map[0] == 1] = 0
            # place in current position
            state_map[0][calculate_coordinates(observation, self.initial_map.shape[1])] = 1
            state_maps.append(state_map)
            char_map = self.deone_hot_map(state_map)
            char_maps.append(char_map)
        return char_maps, state_maps

    def squeeze_map(self):
        squeezed_map = []
        for row in self.initial_map:
            squeezed_row = ""
            for entry in row:
                squeezed_row += entry
            squeezed_map.append(squeezed_row)
        return squeezed_map

def calculate_coordinates(position, n_rows):
    y = (position / n_rows) - (position % n_rows / n_rows)
    x = position - y * n_rows
    return int(y), int(x)

def create_empty_map(x,y):
    new_map = np.zeros((y, x), dtype='U1')
    for iy in range(y):
        for ix in range(x):
            new_map[iy][ix] = 'F'
    return new_map