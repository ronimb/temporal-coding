import matplotlib; matplotlib.use('module://backend_interagg')
from Tempotron import Tempotron
import tensorflow as tf
from make_test_samples import make_test_samples
import numpy as np
from matplotlib import pyplot as plt


def generate_and_train(_sample_params, _tempotron_params, _train_params, verbose=True):
    test_data = make_test_samples(**_sample_params)
    tmp = Tempotron(**_tempotron_params)
    tempotron_input = (test_data['data'], test_data['labels'])

    acc_before = tmp.accuracy(*tempotron_input)
    _training_progress = tmp.train(*tempotron_input, **_train_params)
    max_acc = _training_progress['correct'].max()
    final_acc = _training_progress['correct'][-1][0]
    if verbose:
        print(f'Accuracy before: {acc_before}')
        print(f"Max accuracy: {max_acc:.3f} | Accuracy at final trial: {final_acc:.3f}")
    return _training_progress, test_data, tmp


def check_batch_improvement(test_data, batch_number, _tempotron_params, _training_progress):
    inds = _training_progress['batch_inds'][batch_number]
    batch_trains = test_data['data'][inds]
    batch_labels = test_data['labels'][inds]

    weight_pre = _training_progress['w'][batch_number]
    weight_upd = _training_progress['w_upd'][batch_number]
    weight_post = _training_progress['w_post'][batch_number]

    verified = np.isclose(weight_post, (weight_pre + weight_upd)).all()
    if not verified:
        print('Weights not verified!')
        return None

    w_pre = tf.Variable(weight_pre[np.newaxis, :])
    tmp_pre = Tempotron(**_tempotron_params, weights=w_pre)
    acc_before = tmp_pre.accuracy(batch_trains, batch_labels)

    w_post = tf.Variable(weight_post)
    tmp_post = Tempotron(**_tempotron_params, weights=w_post)
    acc_post = tmp_post.accuracy(batch_trains, batch_labels)

    acc_gain = acc_post - acc_before
    return acc_gain


if __name__ == '__main__':
    sample_params = dict(
        rate=50,
        duration_sec=0.5,  # Given in seconds
        num_neur=10,
        set1_size=500,
        set2_size=500)
    duration_ms = sample_params['duration_sec'] * 1000
    tempotron_params = dict(
        threshold_value=1.5,
        learning_rate=0.001,
        input_shape=(sample_params['num_neur'], duration_ms))
    train_params = dict(
        batch_size=50,
        train_mode='batch',
        iter_num=1000,
        return_params=True)
    training_progress, test_data, _ = generate_and_train(
        sample_params, tempotron_params, train_params, verbose=True)
    print(check_batch_improvement(test_data, 0, tempotron_params, training_progress))



