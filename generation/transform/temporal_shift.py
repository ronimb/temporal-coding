# This file includes all the functions used for simple temporal shifting of stimuli
import numpy as np


def forward_shift(stimulus: np.array, stimulus_duration: float,
                  max_temporal_shift: float, num_transformed: int = 1) -> np.array:
    """
    Transforms stimuli by stochastically shifting all spike times in all neurons
    forward by up to max_temporal_shift seconds

    :param stimulus: A numpy array representing the stimulus in which each line (or object) is a neuron with spike times given in ms
    :param stimulus_duration: Maximal duration of the stimulus, units: ms
    :param max_temporal_shift: The maximal temporal shift by which each spike can be moved, units: ms
    :param num_transformed: Number of shifted versions of the stimulus to generate, units: Integer

    :return: shifted_stimuli: numpy array where each object is a temporally shifted version of the original stimulus
    """

    def _neuron_fwd_shift(orig_neuron: np.array) -> np.array:
        """
        Shifts all spikes of a single neuron in the stimulus forward
        :param orig_neuron: the neuron to be shifted
        :return: new_neuron: temporally shifted version of the neuron
        """

        num_spikes = orig_neuron.size  # Number of spikes in the current neuron

        # Generate uniform values by which to shift
        shift_values = np.random.uniform(low=0,
                                         high=max_temporal_shift,  # Multiplying stimulus_duration to get ms
                                         size=num_spikes)

        # Add shift values to original neurons spike times
        new_neuron = orig_neuron + shift_values
        # Check for spikes which might have exceeded the maximal stimulus stimulus_duration
        exceed_duration = new_neuron > stimulus_duration  # Multiplying stimulus_duration to get ms
        # Making sure spike times remain within the range specified by stimulus_duration
        if exceed_duration.any():
            # Remove exceeding spikes from shifted neuron
            new_neuron = new_neuron[~exceed_duration]
            # Obtain original spike times of exceeding spikes from original neuron
            exceeding_spikes = orig_neuron[exceed_duration]
            # Re-Shift the exceeding spikes by an interval truncated so that it can't exceed the maximal stimulus_duration
            new_spikes = []  # Placeholder for new spike times
            for spike in exceeding_spikes:
                truncated_max_shift = stimulus_duration - spike  # Compute maximal temporal shift for spike
                # Shift spike by up to truncated_max_shift
                spike_shift = np.random.uniform(low=0,
                                                high=truncated_max_shift,
                                                size=1)
                new_spikes.append(spike + spike_shift)  # Add newly shifted spike to list of new spikes
            # Add corrected spikes to the shifted neuron
            new_neuron = np.append(new_neuron, new_spikes)
            # Sorting to account for possible order breaking when Re-Shifting the exceeding spikes
            new_neuron.sort()
        return new_neuron

    shifted_stimuli = []  # Placeholder list for all shifted stimuli
    for i in range(num_transformed):
        shifted_stimulus = []  # Placeholder list for all neurons in current shifted stimulus
        for neuron in stimulus:
            # Create temporally shifted version of current neuron
            shifted_neuron = _neuron_fwd_shift(neuron)
            # Add temporally shifted neuron to new (shifted) stimulus
            shifted_stimulus.append(shifted_neuron)
        # Add shifted stimulus to list of shifted stimuli
        shifted_stimuli.append(shifted_stimulus)
    # Fixing output shape for single shift, convert to numpy array for easier indexing
    if num_transformed == 1:
        shifted_stimuli = np.array(shifted_stimuli)[0]
    else:
        shifted_stimuli = np.array(shifted_stimuli)
    return shifted_stimuli


def symmetric_interval_shift(stimulus: np.array, stimulus_duration: float,
                             interval: tuple, num_transformed: int = 1) -> np.array:
    """
    Transforms stimuli by stochastically shifting all spike times either forward or backward
    within an interval specified in milliseconds.
    For example, specifying the interval (3, 5) will shift spikes either between -5 and -3 ms
    or between 3 and 5 ms.

    :param stimulus: A numpy array representing the stimulus in which each line (or object) is a neuron with spike times given in ms
    :param stimulus_duration: Maximal duration of the stimulus, units: ms
    :param interval: A tuple specifying the symmetric interval by which spikes are temporally shifted, see example. units: ms
    :param num_transformed: Number of shifted versions of the stimulus to generate, units: Integer

    :return: shifted_stimuli: numpy array where each object is a temporally shifted version of the original stimulus
    """

    # %%
    def _correct_start_spikes(orig_spike_time: float) -> float:
        """

        :param orig_spike_time: Time during which the erroneusly shifted spiked occured in the original neuron
        :return: new_shifted_spike: new spike time shifted so that it stays within the interval between t=0 and the maximal stimulus_duration
        """
        # Check whether this spike occurs closer to t=0 than the minimal backward shift
        too_close_to_zero = (orig_spike_time - interval[0]) < 0
        # If the spike is too close to zero, only forward shifting is possible
        if too_close_to_zero:
            spike_shift = np.random.uniform(low=interval[0],
                                            high=interval[1],
                                            size=1)
        # If the spike is not too close to zero, allow either forward shifting or
        # clipped backward shifting, with equal probability
        else:
            # Randomly decide whether to shift forward or backward
            shift_backward = np.random.rand() <= .5
            if shift_backward:
                # Set maximal backward shift so that spike can't move farther back than its original time
                maximal_backward_shift = orig_spike_time
                spike_shift = np.random.uniform(low=interval[0],
                                                high=maximal_backward_shift,
                                                size=1) * -1  # Multiply by -1 to shift backward
            else:
                # In this case, regular forward shifting is possible
                spike_shift = np.random.uniform(low=interval[0],
                                                high=interval[1],
                                                size=1)
        new_shifted_spike = orig_spike_time + spike_shift
        return new_shifted_spike

    # %%
    def _correct_end_spikes(orig_spike_time: float) -> float:
        # Check whether this spike occurs too close to the maximal stimulus_duration for forward shifting
        too_close_to_max_duration = (stimulus_duration - orig_spike_time) < (interval[0] * 1000)
        # If the spike is too close to the maximal stimulus_duration, only backward shifting is possible
        if too_close_to_max_duration:
            spike_shift = np.random.uniform(low=interval[0],
                                            high=interval[1],
                                            size=1) * -1  # Multiply by -1 to shift backward
        # If the spike is sufficiently far from the maximal stimulus_duration, allow either backward shifting or
        # clipped forward shifting with equal probability
        else:
            # Randomly decide whether to shift forward or backward
            shift_backward = np.random.rand() <= .5
            if shift_backward:
                # In this case, regular backward shifting is possible
                spike_shift = np.random.uniform(low=interval[0],
                                                high=interval[1],
                                                size=1) * -1  # Multiply by -1 to shift backward
            else:
                # Set maximal forward shift so that spike can't move forward past the maximal stimulus_duration
                maximal_forward_shift = stimulus_duration - orig_spike_time
                spike_shift = np.random.uniform(low=interval[0],
                                                high=maximal_forward_shift)
        new_shifted_spike = orig_spike_time + spike_shift
        return new_shifted_spike

    # %%
    def _neuron_interval_shift(orig_neuron):
        """
        Shifts all spikes of a single neuron in the stimulus according to the specified interval
        :param orig_neuron: the neuron to be shifted
        :return: new_neuron: temporally shifted version of the neuron
        """

        num_spikes = orig_neuron.size  # Number of spikes in the current neuron

        # Generating random shifts within the interval
        shift_values = np.random.uniform(low=interval[0],  # Multiplying to convert to ms
                                         high=interval[1],  # Multiplying to convert to ms
                                         size=num_spikes)

        # Choose sign for each shift (whether to shift spike forward or backward in time)
        signs = np.random.choice([-1, 1], size=num_spikes)
        # Multiplying to obtain symmetric shifts
        shift_values *= signs
        # Add shift values to original neurons spike times
        new_neuron = orig_neuron + shift_values
        # Check for spikes which exceed either the maximal stimulus stimulus_duration or occur before t=0
        outside_range = (new_neuron < 0) | (new_neuron > stimulus_duration)
        # Making sure spike times remain between 0 and stimulus_duration
        if outside_range.any():
            # Obtain the erroneous spikes themselves
            spikes_outside_range = new_neuron[outside_range]
            # Obtain original spike times of erroneus spikes
            original_spikes = (orig_neuron)[outside_range]
            # Re-shift the erroneous spikes by clipping the interval
            new_spikes = []  # Placeholder for new spike times
            for err_spike, orig_spike in zip(spikes_outside_range, original_spikes):
                # Corrective action for spikes before t=0
                if np.sign(err_spike) == -1:
                    new_spike = _correct_start_spikes(orig_spike)

                # Corrective action for spikes after maximal stimulus_duration
                elif np.sign(err_spike) == 1:
                    new_spike = _correct_end_spikes(orig_spike)

                new_spikes.append(new_spike)  # Add newly shifted spike to list of new spikes
            # Remove erroneus spikes from shifted neuron
            new_neuron = new_neuron[~outside_range]
            # Add corrected spikes to the shifted neuron
            new_neuron = np.append(new_neuron, new_spikes)
        # Sorting to fix order of spikes
        new_neuron.sort()
        return new_neuron

    # %%
    shifted_stimuli = []  # Placeholder list for all shifted stimuli
    for i in range(num_transformed):
        shifted_stimulus = []  # Placeholder list for anew_spikes.append(spike + spike_shift)  # Add newly shifted spike to list of new spikesll neurons in current shifted stimulus
        for neuron in stimulus:
            # Create temporally shifted version of current neuron
            shifted_neuron = _neuron_interval_shift(neuron)
            # Add temporally shifted neuron to new (shifted) stimulus
            shifted_stimulus.append(shifted_neuron)
        # Add shifted stimulus to list of shifted stimuli
        shifted_stimuli.append(shifted_stimulus)
    # Fixing output shape for single shift, convert to numpy array for easier indexing
    if num_transformed == 1:
        shifted_stimuli = np.array(shifted_stimuli)[0]
    else:
        shifted_stimuli = np.array(shifted_stimuli)
    # %%
    return shifted_stimuli
