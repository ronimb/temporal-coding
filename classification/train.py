import numpy as np
from . import Tempotron
from generation.set_classes import StimuliSet
from generation.conversion import convert_stimuli_set


# %%
def batch_train(stimuli_set: StimuliSet,
                learning_rate: float, batch_size: int, training_repetitions: int,
                tempotron_tau: float, tempotron_threshold: float,
                tempotron: Tempotron = None):
    # Handle stimuli set conversion if needed (SLOWS PERFORMANCE IF CONVERSION NEEDED)
    if not stimuli_set.converted:  # In this case we have a Normal stimuli set and must convert
        convert_stimuli_set(stimuli_set)

    # Extract number of neurons - This assumes stimuli of Converted type
    number_of_neurons = int(np.unique(stimuli_set.stimuli['index']).shape[0] / len(stimuli_set))

    # If no Tempotron object was passed for classification, create one and return it in the end
    if not Tempotron:
        tempotron = Tempotron(number_of_neurons=number_of_neurons,
                              tau=tempotron_tau,
                              threshold=tempotron_threshold,
                              stimulus_duration=stimuli_set.stimulus_duration)
        return_tempotron = True

    # Train
    tempotron.train(
        stimuli=stimuli_set.stimuli,
        labels=stimuli_set.labels,
        batch_size=batch_size,
        num_reps=training_repetitions,
        learning_rate=learning_rate
    )
    if return_tempotron:
        return tempotron
