import matplotlib; matplotlib.use('module://backend_interagg')
from Tempotron import Tempotron
from make_test_samples import make_test_samples
from matplotlib import pyplot as plt


sample_params = dict(rate=50,
                     duration_sec=0.5,  # Given in seconds
                     num_neur=10,
                     set1_size=500,
                     set2_size=500)
test_data = make_test_samples(**sample_params)

duration_ms = sample_params['duration_sec'] * 1000
tempotron_params = dict(threshold_value=1.5,
                        learning_rate=0.001,
                        input_shape=(sample_params['num_neur'], duration_ms))
tmp = Tempotron(**tempotron_params)

acc = []
for _ in range(30):
    tmp.initvals()
    tmp.train(test_data['data'], test_data['labels'])
    acc.append(tmp.accuracy(test_data['data'], test_data['labels']))



