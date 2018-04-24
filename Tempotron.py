import numpy as np
import brian2 as b2

from make_test_samples import make_test_samples


def train():
    pass


if __name__ == '__main__':
    test_sample_params = dict(rate=50,
                              duration_sec=1,
                              num_neur=10,
                              shift_size=5,
                              set1_size=500,
                              set2_size=500)
    samples = make_test_samples(**test_sample_params)
