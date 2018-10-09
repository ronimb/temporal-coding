import numpy as np
from . import Tempotron
from generation.set_classes import StimuliSet
from generation.conversion import convert_stimuli_set


# %%
def evaluate(stimuli_set: StimuliSet,
             tempotron_tau: float = None, tempotron_threshold: float = None,
             tempotron: Tempotron = None, conversion_pool_size: int=8) -> float:
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
    # Converting to format requierd for processing in brian module
    if not hasattr(stimuli_set, '_tempotron_converted_stimuli'): #handling conversion-on-demand
        stimuli_set._make_tempotron_converted(pool_size=conversion_pool_size)
    tempotron_converted_stimuli = stimuli_set._tempotron_converted_stimuli

    # Extract number of neurons
    number_of_neurons = stimuli_set.stimuli.shape[0]

    # If no Tempotron object was passed for classification, create one
    if not tempotron:
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
        stimuli=tempotron_converted_stimuli,
        labels=stimuli_set.labels,
    )
    # Average results for mean accuracy over entire set
    mean_success = np.mean(correct_classifications)
    return mean_success
