from generation.make_stimuli_set import make_set_from_specs
from generation import transform
from classification import evaluate, batch_train
from classification import Tempotron
from time import time
# %%
set_size = 200

creation_params = dict(
    frequency=15,
    number_of_neurons=30,
    stimulus_duration=500,
)

set_transform_function = transform.stochastic_release
set_transform_params = dict(
    release_duration=5,
    number_of_vesicles=20,
    stimulus_duration=creation_params['stimulus_duration'],
    release_probability=1,
    num_transformed=50
)

origin_transform_function = transform.symmetric_interval_shift
origin_transform_params = dict(
    stimulus_duration=creation_params['stimulus_duration'],
    interval=(3, 7)
)

tempotron_params = dict(
    number_of_neurons=creation_params['number_of_neurons'],
    tau=2,
    threshold=0.05,
    stimulus_duration=creation_params['stimulus_duration']
)

training_params = dict(
    learning_rate=1e-3,
    batch_size=50,
    training_repetitions=15,
    fraction=0.21,
)
# %%
stimuli_set = make_set_from_specs(**creation_params, set_size=set_size,
                                   set_transform_function=set_transform_function,
                                   set_transform_params=set_transform_params)

T = Tempotron(**tempotron_params)

t1 = time()
pre = evaluate(stimuli_set, tempotron=T)
batch_train(stimuli_set, tempotron=T, **training_params)
post = evaluate(stimuli_set, tempotron=T)
d1 = time() - t1
# %%
stimuli_set._make_tempotron_converted()

T = Tempotron(**tempotron_params)

t2 = time()
pre = evaluate(stimuli_set, tempotron=T)
batch_train(stimuli_set, tempotron=T, **training_params)
post = evaluate(stimuli_set, tempotron=T)
d2 = time() - t2

print(d1)
print(d2)