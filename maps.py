import numpy as np

map_ten_x_ten_test_training = np.array([ ['H', 'F', 'H', 'F', 'F', 'S', 'F', 'H', 'F', 'H'],
                                         ['H', 'F', 'H', 'F', 'F', 'H', 'H', 'F', 'H', 'F'],
                                         ['F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                                         ['F', 'F', 'H', 'F', 'H', 'H', 'F', 'F', 'F', 'F'],
                                         ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'H', 'F', 'H'],
                                         ['F', 'F', 'F', 'H', 'F', 'H', 'F', 'F', 'F', 'F'],
                                         ['F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'H', 'F'],
                                         ['H', 'F', 'H', 'H', 'H', 'F', 'F', 'H', 'F', 'H'],
                                         ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                                         ['H', 'F', 'G', 'F', 'F', 'F', 'F', 'F', 'F', 'F']]
                        )

# for testing only
unsolvable = np.array([                  ['H', 'F', 'H', 'F', 'H', 'S', 'H', 'H', 'F', 'H'],
                                         ['H', 'F', 'H', 'F', 'H', 'H', 'H', 'F', 'H', 'F'],
                                         ['F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                                         ['F', 'F', 'H', 'F', 'H', 'H', 'F', 'F', 'F', 'F'],
                                         ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'H', 'F', 'H'],
                                         ['F', 'F', 'F', 'H', 'F', 'H', 'F', 'F', 'F', 'F'],
                                         ['F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'H', 'F'],
                                         ['H', 'F', 'H', 'H', 'H', 'F', 'F', 'H', 'F', 'H'],
                                         ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                                         ['H', 'F', 'G', 'F', 'F', 'F', 'F', 'F', 'F', 'F']]
                        )

initial_test = np.array([['S', 'F', 'F', 'F'], ['F', 'H', 'F', 'H'], ['F', 'F', 'F', 'H'], ['H', 'F', 'F', 'G']])