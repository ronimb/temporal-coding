import attr
import numpy as np
from multiprocessing import Pool
from numba import jit, prange
from tools import calc_stimuli_distance
import pickle


# ToDo: implement loading functions
# %% Helper functions
def _numpy_arr_convert(neurons):
    """
    used only due to possible cases where calling np.array on a list of neurons with similar length
    results in a 2d numpy array instead of a 1d numpy array of Neurons
    :param neurons:
    :return:
    """
    # If already numpy array, do nothing
    if isinstance(neurons, np.ndarray):
        return neurons

    # if not, make sure its either a list or a tuple, and then manually create the array
    elif isinstance(neurons, (list, tuple)):
        neurons_array = np.empty(len(neurons), dtype=object)
        for i, neuron in enumerate(neurons):
            neurons_array[i] = neuron
        return neurons_array


def _bool_poisson(frequency: int, num_neurons: int, stimulus_duration: float, dt: float = 1e-5) -> np.array:
    """
    Intended for internal use only, generates collection of poisson neurons and returns
    a boolean array of spikes according to the specified stimulus_duration and dt.

    :param frequency: The average firing frequency of each neuron in the sample, units: Hz
    :param num_neurons: Number of neurons in the stimulus, units: Integer
    :param stimulus_duration: Maximal stimulus_duration of the stimulus, units: ms
    :param dt: Simulation time step, units: Sec
    """
    duration_sec = stimulus_duration / 1000
    num_bins = np.round(duration_sec / dt).astype(int)  # Number of time bins in the simulation

    # Generate uniform random values between 0-1 in each time bin for each neuron
    random_vals = np.random.rand(num_neurons, num_bins)

    # Insert spike wherever the random values is smaller then frequency * dt
    bool_spikes = random_vals <= (frequency * dt)  # As boolean array

    return bool_spikes


@jit(parallel=True)
def _tempotron_format_convert(stimulus: object) -> np.array:
    """
    This function converts the events from a Stimulus to a structured
    numpy array of events, where events that occur very close to each other in time
    (such as successive vesicle releases) are grouped and counted.
    Events are also separated according to neurons by their indices.
    The array contains the following fields:
        index:  Represents the index of the neuron at which the event or events occured
        time:   Denotes the time at which the event(s) occured
        count:  Denotes the number of events that occured at this time

    :param stimulus: Stimulus object instance representing collection of events from several neurons
    :return: converted_neurons: A structured numpy array with the following fields:
    """
    # create placeholder lists
    indexes = []
    times = []
    counts = []
    for neuron_index in prange(stimulus.size):
        neuron_events = np.trunc(stimulus[neuron_index].events * 10) / 10
        event_times, event_counts = np.unique(neuron_events, return_counts=True)
        indexes.extend([neuron_index] * event_times.shape[0])
        times.extend(event_times)
        counts.extend(event_counts)

    converted_neurons = np.array(
        np.zeros(len(times)),
        dtype=dict(
            names=('index', 'time', 'count'),
            formats=(int, float, int)
        ))
    converted_neurons['index'] = indexes
    converted_neurons['time'] = times
    converted_neurons['count'] = counts
    return converted_neurons


# %% Classes
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

    # Used to check the number of stimuli in the set
    def __len__(self):
        return len(self.labels)

    @property  # Set as property to call without brackets
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
        # Combine labels
        labels = np.array([*self.labels, *other.labels])
        # Check if only two unique labels exist, and if so calculate distance
        combined_stimuli_set = StimuliSet(
            stimuli=combined_stimuli,
            labels=labels,
            stimulus_duration=self.stimulus_duration,
            original_stimuli=(self.original_stimuli, other.original_stimuli),
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
        self.stimuli = self.stimuli[randomized_indexes]

    def select_subset(self):
        subset = None
        return subset

    def convert(self):
        pass

    def split(self):
        pass

    def save(self):
        pass


# %%
@attr.s
class Neuron:
    events = attr.ib(converter=np.array)
    frequency_generated = attr.ib()
    stimulus_duration = attr.ib()
    event_type = attr.ib(default='spikes')  # 'spikes', 'vesicles' or other custom made
    frequency_actual = attr.ib()  # Is set automatically upon instantiation

    # Calculate real frequency
    @frequency_actual.default  # Using @ decorator to set default value for property
    def _calc_actual_frequency(self):
        return self.size * (1000 / self.stimulus_duration)

    def __len__(self):
        return self.events.shape[0]

    @property  # Set as property to call without brackets
    def size(self):
        return len(self)

    def __getitem__(self, item):
        return self.events[item]  # Relying on numpy indexing

    def __setitem__(self, key, value):
        self.events[key] = value  # Relying on numpy indexing


# %%
@attr.s
class Stimulus:
    neurons = attr.ib(converter=_numpy_arr_convert)  # Must be a numpy array of Neuron objects
    frequency_generated = attr.ib()
    stimulus_duration = attr.ib()
    event_type = attr.ib(default='spikes')  # 'spikes', 'vesicles' or other custom made
    frequency_average = attr.ib()  # Is set automatically upon instantiation

    @frequency_average.default  # Using @ decorator to set default value for property
    def calc_average_frequency(self):
        if self.event_type == 'spikes':
            neurons_actual_frequencies = [neuron.frequency_actual for neuron in self.neurons]
            return np.mean(neurons_actual_frequencies)
        else:
            return None

    def __len__(self):
        """
        Returns the number of neurons in the stimulus
        :return:
        """
        return len(self.neurons)

    @property  # Set as property to call without brackets
    def size(self):  # Simple alias for __len__
        return len(self)

    # Set up indexing neuron at Stimulus.neurons may be directly accessed through Stimulus[index]
    def __getitem__(self, neuron_indexes):
        return self.neurons[neuron_indexes]  # Relying on numpy indexing

    def __setitem__(self, neuron_indexes, values):
        self.neurons[neuron_indexes] = values  # Relying on numpy indexing

    # Conversion method to get stimulus in a format that can be fed to the brian based Tempotron
    def _tempotron_format_convert(self):
        return _tempotron_format_convert(self)

    def calculate_distance_from(self, other):
        """
        This method computes the average distance between neurons in two stimuli
        using the spike-distance metric  (see: http://www.scholarpedia.org/article/SPIKE-distance)

        :param other: Another stimulus object with the same duration and number of neurons
        """
        # Verify stimuli are comparable
        return calc_stimuli_distance(self, other)

    # Wrapper for pickle to make saving easier
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    ## REFRACTORY PERIOD NOT YET IMPLEMENTED!
    def make(frequency: int, number_of_neurons: int, stimulus_duration: float, refractory_period: float = 2e-3,
             dt: float = 1e-5, exact_frequency=False) -> np.array:
        """
        Used to generate a new Stimulus object with the specified parameters for each neuron

        :param frequency: The average firing frequency of each neuron in the sample, units: Hz
        :param number_of_neurons: Number of neruons in the stimulus, units: Integer
        :param stimulus_duration: Maximal stimulus_duration of the stimulus, units: ms
        :param refractory_period: Length of minimal period between two spikes, units: Sec  CURRENTLY NOT IMPLEMENTED
        :param dt: Simulation time step, units: Sec
        :param exact_frequency: whether all neurons fire with the same exact frequency, or the same average frequency
        :return:
        """
        def _return_exact(bool_stimulus: np.array) -> np.array:
            """
            Used for filtering the stimulus for neurons firing only at the exact frequency
            :param bool_stimulus: stimulus to be filtered
            :return: exact: filtered boolean stimulus
            """
            duration_sec = stimulus_duration / 1000
            # Count spikes in each neuron
            spike_count = bool_stimulus.sum(1)
            # Find neurons firing at the correct frequency
            correct_count = spike_count == np.round((frequency * duration_sec))  # Rounding to handle edge cases
            # Keep only those neurons firing at the correct frequency
            exact = bool_stimulus[correct_count]
            return exact

        # Generate the stimulus in boolean form
        spikes_bool = _bool_poisson(frequency, number_of_neurons, stimulus_duration, dt)
        # Check that each neuron spikes at least once and re-generate otherwise
        num_spikes = spikes_bool.sum(1)
        zero_spikes = num_spikes == 0
        while zero_spikes.any():
            num_zero_spikes = zero_spikes.sum()  # Count  number of neurons with no spikes
            new_neurons = _bool_poisson(frequency, num_zero_spikes, stimulus_duration)  # Generate new neurons
            spikes_bool[zero_spikes] = new_neurons
            # Check again
            num_spikes = spikes_bool.sum(1)
            zero_spikes = num_spikes == 0

        # Handle exact frequency requirement
        if exact_frequency:
            # Filter out neurons not firing at the exact frequency
            spikes_bool = _return_exact(spikes_bool)
            # Generate new neurons until we have the desired number of neurons firing at the exact frequency
            while spikes_bool.shape[0] < number_of_neurons:
                # Generate new neurons
                new_neurons_bool = _bool_poisson(frequency, number_of_neurons * 2, stimulus_duration, dt)
                # Filter these new neurons
                new_neurons_bool = _return_exact(new_neurons_bool)
                # Add these correct neurons to the stimulus
                spikes_bool = np.append(spikes_bool, new_neurons_bool, 0)
            # Making sure we have precisely the desired number of neurons and no more
            spikes_bool = spikes_bool[0:number_of_neurons, :]

        # Transforming the boolean stimulus to an array of Neuron objects
        neuron_index, firing_indexes = np.where(spikes_bool)  # Find indexes of neurons and indexes of spikes
        times = firing_indexes * dt * 1000  # Transform firing time indexes to seconds

        neuron_array = [Neuron(events=times[neuron_index == i],
                               frequency_generated=frequency,
                               stimulus_duration=stimulus_duration,
                               event_type='spikes') for i in range(number_of_neurons)]

        # Make Stimulus object
        stimulus = Stimulus(neurons=neuron_array,
                            frequency_generated=frequency,
                            stimulus_duration=stimulus_duration,
                            event_type='spikes')
        return stimulus
