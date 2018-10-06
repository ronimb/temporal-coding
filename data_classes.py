"""
Definitions for stimuli_set objects
"""
import attr
import numpy as np
from multiprocessing import Pool
from numba import jit, prange
from tools import calc_stimuli_distance


# %%
@attr.s
class StimuliSet:
    """
    This is a class intended to hold a collection of stimuli for use in the experiments,
    it has the following two mandatory fields:
        stimuli - A collection of stimuli in one of two formats:
                        Normal:     A numpy object array where each item is a stimulus as an array of neurons
                                    and their respective event times
                        Converted:  A structured numpy array with the following fields: index, time, count
                                    where each item represents events taking place at a 'time',
                                    originating from neuron #'index' in the set,
                                    and with 'count' representing the number of events that took place at that junction.
                                    count is required to account for numerous vesicles released in short time intervals
        labels - Label for each stimulus in the set according to its origin (from one of two possible original stimuli)
        stimulus_duration - Maximal duration of the stimulus, units: ms

    The following fields are optional:
        original_stimuli -  tuple containing both original stimuli as numpy arrays of neurons and their
                            corresponding event times (spikes or vesicle releases)
        original_stimuli_distance - The average spike-distance metric between neurons in the two stimuli

    In addition, the field "converted" specifies whether or not the stimuli in the object are of a normal or
    converted format

    """
    # Initializing class attributes
    stimuli = attr.ib()  # Contains all the stimuli, in either normal or converted format
    labels = attr.ib(converter=np.array)  # Contains the labels for each of the stimuli
    stimulus_duration = attr.ib()  # The maximal duration of the original stimulus
    original_stimuli = attr.ib(
        default=None)  # Can be used to contain the original stimuli from which the set was generated
    original_stimuli_distance = attr.ib(
        default=None)  # Can be used to contain the distance between the originating stimuli of a set
    converted = attr.ib(default=False)  # Wheteher the stimuli in the set are converted or not

    # Used to check the number of stimuli in the set
    def __len__(self):
        return len(self.labels)

    def size(self):  # Simple alias
        return len(self)

    def __getitem__(self, varargin):
        pass

    def __setitem__(self, key, value):
        pass

    def __add__(self, other):
        """
        Used to join two StimuliSet's together into one set

        :param other: another StimuliSet object of any size
        :return: combined_stimuli_set: A StimuliSet made from both sets combined
        """

        # Calculate total size
        new_size = len(self) + len(other)
        # Combine stimuli
        combined_stimuli = np.array([*self.stimuli,
                                     *other.stimuli])
        # Handle for converted stimuli
        if self.converted:
            number_of_converted_indexes = self.stimuli.shape[0]
            combined_stimuli['index'][number_of_converted_indexes:] += number_of_converted_indexes

        # Combine labels
        labels = np.array([*self.labels, *other.labels])
        # Check if only two unique labels exist, and if so calculate distance
        combined_stimuli_set = StimuliSet(
            stimuli=combined_stimuli,
            labels=labels,
            stimulus_duration=self.stimulus_duration,
            original_stimuli=(self.original_stimuli, other.original_stimuli),
            converted=self.converted
        )
        return combined_stimuli_set

    def shuffle(self):
        """
        This method is used to take a stimuli)set of stimuli shuffle their order in the list to
        randomize the order obtained by serial generation.
        """
        number_of_stimuli = len(self)
        randomized_indexes = np.random.permutation(number_of_stimuli)
        # Shuffle labels
        self.labels = self.labels[randomized_indexes]
        # Shuffle stimuli
        if self.converted:  # Need to handle indexes in converted stimuli slightly differently
            # Calculating number of neurons
            number_of_neurons = np.unique(self.stimuli['index']).shape[0] / number_of_stimuli
            # using the modules operator to find the original index of the stimulus with the neuron index
            stimulus_indexes = self.stimuli['index'] % number_of_stimuli
            neuron_indexes = self.stimuli['index'] % number_of_neurons
            for original_stimulus_index, new_stimulus_index in zip(range(number_of_stimuli), randomized_indexes):
                # Find location of indexes associated with stimulus #original_index
                current_stimulus_indexes = [stimulus_indexes == original_stimulus_index]
                new_indexes = (new_stimulus_index * number_of_neurons) + neuron_indexes[current_stimulus_indexes]
                self.stimuli['index'] = new_indexes

        else:
            self.stimuli = self.stimuli[randomized_indexes]

    def select_subset(self):
        subset = None
        return subset

    def convert(self):
        pass

    def split(self):
        pass

    @staticmethod
    def load():
        print('loaeded be keilu')

    @staticmethod
    def make_from_specifications():
        pass

    @staticmethod
    def make_from_stimuli():
        pass


# %%
@attr.s
class Stimulus:
    converted = attr.ib()
    average_frequency = attr.ib()

    @average_frequency.default
    def calc_average_frequency(self):
        # calc_average_frequency()
        pass

    def __len__(self):
        return

    def convert(self):
        return


# %%
@attr.s
class Neuron:
    events = attr.ib(converter=np.array)
    frequency_generated = attr.ib()
    stimulus_duration = attr.ib()
    frequency_actual = attr.ib()
    event_types = attr.ib(default='spikes')

    # Calculate real frequency
    @frequency_actual.default
    def _calc_actual_frequency(self):
        return self.size() * (1000 / self.stimulus_duration)

    def __len__(self):
        return self.events.shape[0]

    def size(self):
        return len(self)

