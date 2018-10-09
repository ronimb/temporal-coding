import numpy as np
from . import Tempotron
from generation.set_classes import StimuliSet
from generation.conversion import convert_stimuli_set


# %%
def batch_train(stimuli_set: StimuliSet,
                learning_rate: float, batch_size: int, training_repetitions: int,
                tempotron_tau: float = None, tempotron_threshold: float = None,
                tempotron: Tempotron = None,
                conversion_pool_size: int = 8, **kwargs):
    if any(kwargs):  #Ignore
        pass
    # Converting to format requierd for processing in brian module
    if not hasattr(stimuli_set, '_tempotron_converted_stimuli'):  # handling conversion-on-demand
        stimuli_set._make_tempotron_converted(pool_size=conversion_pool_size)
    tempotron_converted_stimuli = stimuli_set._tempotron_converted_stimuli

    # Extract number of neurons
    number_of_neurons = stimuli_set.stimuli.shape[0]

    # If no Tempotron object was passed for classification, create one and return it in the end
    if not tempotron:
        tempotron = Tempotron(number_of_neurons=number_of_neurons,
                              tau=tempotron_tau,
                              threshold=tempotron_threshold,
                              stimulus_duration=stimuli_set.stimulus_duration)
        return_tempotron = True
    else:
        return_tempotron = False
    # Train
    tempotron.train(
        stimuli=tempotron_converted_stimuli,
        labels=stimuli_set.labels,
        batch_size=batch_size,
        num_reps=training_repetitions,
        learning_rate=learning_rate
    )
    if return_tempotron:
        return tempotron
