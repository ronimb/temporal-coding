"""
Definitions for stimuli_set objects
"""
import attr
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
                                    where each item represents events taking place at a 'time' from neuron #'index'
                                    in the set, and with 'count' representing the number of events that
                                    took place at that junction.
                                    count is required to account for numerous vesicles released in short time intervals
        labels - Label for each stimulus in the set according to its origin (from one of two possible original stimuli)

    The following fields are optional:
        original_stimuli -  tuple containing both original stimuli as numpy arrays of neurons and their
                            corresponding event times (spikes or vesicle releases)
        original_stimuli_distance - The average spike-distance metric between neurons in the two stimuli

    In addition, the field "converted" specifies whether or not the stimuli in the object are of a normal or
    converted format

    """
    # Initializing class attributes
    stimuli = attr.ib()  # Contains all the stimuli, in either normal or converted format
    labels = attr.ib()  # Contains the labels for each of the stimuli
    original_stimuli = attr.ib(default=None)  # Can be used to contain the original stimuli from which the set was generated
    original_stimuli_distance = attr.ib(default=None)  # Can be used to contain the distance between the originating stimuli of a set
    converted = attr.ib(default=False)  # Wheteher the stimuli in the set are converted or not

    # Used to check the number of stimuli in the set
    def __len__(self):
        return len(self.labels)