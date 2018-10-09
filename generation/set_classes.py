"""
Definitions for stimuli_set objects
"""
import attr
import numpy as np
from generation.conversion import convert_stimuli_set
from sklearn.model_selection import train_test_split


# %%
@attr.s
class StimuliSet:
    """
    This is a class intended to hold a collection of stimuli for use in the experiments,
    it has the following two mandatory fields:
        stimuli - A collection of stimuli in the following format:
                        A numpy object array where each item is a stimulus as an array of neurons
                        and their respective event times
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
    stimuli = attr.ib()  # Contains all the stimuli
    labels = attr.ib(converter=np.array)  # Contains the labels for each of the stimuli
    stimulus_duration = attr.ib()  # The maximal duration of the original stimulus
    original_stimuli = attr.ib(
        default=None)  # Can be used to contain the original stimuli from which the set was generated
    original_stimuli_distance = attr.ib(
        default=None)  # Can be used to contain the distance between the originating stimuli of a set


    _tempotron_converted_stimuli = attr.ib(init=False)

    def _make_tempotron_converted(self, pool_size=8):
        """
        This is useful in order to not have to go through the conversion process many times,
        this does NOT get saved in the final files, but does get regenerated whenever the experiment is run
        :param pool_size: number of cores to use for conversion process
        :return:
        """
        self._tempotron_converted_stimuli = self.convert_for_tempotron(pool_size=pool_size)

    # Used to check the number of stimuli in the set
    def __len__(self):
        return len(self.labels)

    @property
    def size(self):
        return len(self)

    def convert_for_tempotron(self, pool_size=8):
        return convert_stimuli_set(self, pool_size=pool_size)

    def __getitem__(self, stimulus_indexes):
        """
        this will return a structured numpy array of the desired stimulus indexes with the following fields:
            - stimulus: the stimulus at the index
            - label: the label of the stimulus
        """
        selected_stimuli = self.stimuli[stimulus_indexes]
        selected_labels = self.labels[stimulus_indexes]
        combined_array = np.array(np.zeros(selected_labels.size),
                                  dtype={'names': ('stimuli', 'labels'),
                                         'formats': (object, bool)})
        combined_array['stimuli'] = [selected_stimuli] if (selected_stimuli.ndim == 1) else list(selected_stimuli)
        combined_array['labels'] = selected_labels
        return combined_array

    def __setitem__(self, key, value):
        raise NotImplemented('Setting items not implemented for StimuliSet objects')

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
        self.stimuli = self.stimuli[randomized_indexes]

    def split(self, fraction: float = 0.5):
        """
        splits the stimuli set into two parts, one with "fraction" of the stimuli and the other with "1-fraction",
        handles rounding
        :param fraction: Where to split the StimuliSet
        :return: A tuple with two stimuli sets split from the original set
        """
        # Applying scikit-learns train_test_split function - maints label proportions as well
        stimuli_set_1, stimuli_set_2, labels_set_1, labels_set_2 = train_test_split(
            self.stimuli, self.labels, stratify=self.labels)

        stimuli_set_1 = StimuliSet(stimuli=stimuli_set_1,
                                   labels=labels_set_1,
                                   stimulus_duration=self.stimulus_duration,
                                   original_stimuli=self.original_stimuli,
                                   original_stimuli_distance=self.original_stimuli_distance)
        stimuli_set_2 = StimuliSet(stimuli=stimuli_set_2,
                                   labels=labels_set_2,
                                   stimulus_duration=self.stimulus_duration,
                                   original_stimuli=self.original_stimuli,
                                   original_stimuli_distance=self.original_stimuli_distance)

        return stimuli_set_1, stimuli_set_2
