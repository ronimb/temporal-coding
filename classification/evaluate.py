import numpy as np
from . import Tempotron
from generation.set_classes import StimuliSet
from generation.conversion import convert_stimuli_set


# %%
def evaluate(stimuli_set: StimuliSet,
             tempotron_tau: float = None, tempotron_threshold: float = None,
             tempotron: Tempotron = None) -> float:
    """
    This function uses the Tempotron model to evaluate the accuracy of classification
    over a given stimuli_set.
    It may be used in one of two ways:
    - Directly suppply the the Tempotron object with which to evaluate
    - Specify parameters for ad-hoc creation of Tempotron model (which will then be returned)

    :param stimuli_set:  A StimuliSet object with the AT LEAST the following fields:
                         stimuli -  A collection of stimuli in the following format:
                                    Normal:     A numpy object array where each item is a stimulus as an array of
                                                neurons and their respective event times
                         labels -   Label for each stimulus in the set according to its origin
                                    (from one of two possible original stimuli)
                         stimulus_duration - copy of the stimulus stimulus_duration above
                         converted - Whether the stimuli are Converted or not
    # The next two parameters will only be used if not tempotron object is supplied
    :param tempotron_tau:  Decay constant for voltage in tempotron model
    :param tempotron_threshold: Threshold for firing in the model, a neuron that fires results in a "1" classification
    :param tempotron: (optional) If supplied, will use this tempotron for classification, this facilitates reuse of object
                      for efficiency purposes, as well as testing

    :return mean_success: Average fraction of correct classifications over the entire set
    """
    # Handle stimuli set conversion if needed (SLOWS PERFORMANCE IF CONVERSION NEEDED)
    if not stimuli_set.converted:  # In this case we have a Normal stimuli set and must convert
        convert_stimuli_set(stimuli_set)

    # Extract number of neurons - This assumes stimuli of Converted type
    number_of_neurons = int(np.unique(stimuli_set.stimuli['index']).shape[0] / len(stimuli_set))

    # If no Tempotron object was passed for classification, create one
    if not Tempotron:
        tempotron = Tempotron(number_of_neurons=number_of_neurons,
                              tau=tempotron_tau,
                              threshold=tempotron_threshold,
                              stimulus_duration=stimuli_set.stimulus_duration)
    # Create brian network for classification
    tempotron.make_classification_network(number_of_stimuli=len(stimuli_set),
                                          network_name='evaluation')
    # Classify each stimulus with current weights and return result
    correct_classifications = tempotron.accuracy(
        network_name='evaluation',
        stimuli=stimuli_set.stimuli,
        labels=stimuli_set.labels,
    )
    # Average results for mean accuracy over entire set
    mean_success = np.mean(correct_classifications)
    return mean_success
