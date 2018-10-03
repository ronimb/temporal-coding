"""
This file is used to generate a set of labelled stimuli (each belonging to either category 0 or 1)
according to various rules. Transformation functions from the transform module are used extensively. see comment below.

To view all possible transformation functions (they are the same wheter applied to set or to origin)
simply import the transform module from generation "from generation import transform" and look at the
available functions, or check the functions in the generation.transform modules by viewing the files in that module
"""
# %%
from tools import set_tools, calc_stimuli_distance
from generation import make_stimulus


def make_set_from_specs(frequency, number_of_neurons, stimulus_duration,
                        set_size, set_transform_function, set_transform_params,
                        origin_transform_function=None, origin_transform_params=None,
                        exact_frequency=False, shuffle_set_order=True,
                        return_original_stimuli=False) -> tuple:
    """
    This function accepts specifications for stimulus creation,
    creates two stimuli with these specifications,
    and then applies the set_transform_function with set_transform_params to each
    stimulus, collecting the transformed versions of both stimuli into the stimuli_set.

    if origin_transform_function AND origin_transform_params are specified,
    One original stimulus will be generated with the desired specs,
    whereas the other will be generated by applying origin_transform_function with origin_transform_params
    to that same stimulus.



    :param frequency: The average firing frequency of each neuron in the sample, units: Hz
    :param number_of_neurons:  Number of neurons in the stimulus, units: Integer
    :param stimulus_duration: Maximal duration of the stimuli, units: ms
    :param set_size: number of stimuli in the final set, an equal number is generated from each original stimulus
    :param set_transform_function: A transformation from generation.transform to be applied to the original stimuli
    :param set_transform_params: The parameters to be used with set_transformation_function
    :param origin_transform_function: A transformation from generation.transform to be applied to one original stimulus in order to generate the other
    :param origin_transform_params: The parameters to be used with origin_transform_function
    :param exact_frequency: whether all neurons fire with the same exact frequency, or the same average frequency
    :param shuffle_set_order: Whether to return the set shuffled or ordered by original stimulus
    :param return_original_stimuli: whether to return the two original stimuli or not, default False

    :return stimuli_set: A structured numpy array of the transformed stimuli
                        containing set_size elements, with each element having two fields:
                        'stimulus': contains the transformed stimulus as a numpy array of neurons and event times
                        'label': True if stimulus originated from original_stimulus_a, False if from original_stimulus_b
    :return distance_between_origins: The average spike-distance metric between neurons in the two stimuli
    :return original_stimuli: (optional) tuple containing both original stimuli as numpy arrays of neurons and their
                              corresponding event times (spikes or vesicle releases)
    """
    # Creating at least one original stimulus
    original_stimulus_a = make_stimulus(frequency=frequency, number_of_neurons=number_of_neurons,
                                        stimulus_duration=stimulus_duration, exact_frequency=exact_frequency)
    # Either creating a stimulus by applying origin_transform_function to original_stimulus_a, or creating an entirely
    # New original_stimulus_b
    if origin_transform_function and origin_transform_params:  # Check if both parameters required for transformation were supplied
        # Create original_stimulus_b by transforming original_stimulus_a
        original_stimulus_b = origin_transform_function(stimulus=original_stimulus_a, **origin_transform_params)
    else:
        original_stimulus_b = make_stimulus(frequency=frequency, number_of_neurons=number_of_neurons,
                                            stimulus_duration=stimulus_duration, exact_frequency=exact_frequency)
    # Calculate distance between original stimuli
    distance_between_origins = calc_stimuli_distance(stimulus_a=original_stimulus_a,
                                                     stimulus_b=original_stimulus_b,
                                                     stimulus_duration=stimulus_duration)
    # Determine how many transformed version to create from original_stimulus_a and original_stimulus_b
    num_transformed = int(set_size / 2)
    set_transform_params['num_transformed'] = num_transformed  # Add to parameter dictionary
    # Create transformed versions from both stimuli
    transformed_set_from_a = set_transform_function(stimulus=original_stimulus_a, **set_transform_params)
    transformed_set_from_b = set_transform_function(stimulus=original_stimulus_b, **set_transform_params)

    # Combine, label and also shuffle (or not)
    stimuli_set = set_tools.combine_and_label(set_a=transformed_set_from_a,
                                              set_b=transformed_set_from_b,
                                              shuffle=shuffle_set_order)
    # Handle returning of original stimuli
    if return_original_stimuli:
        original_stimuli =(original_stimulus_a, original_stimulus_b)
    else:
        return stimuli_set, distance_between_origins


def make_set_from_stimuli(original_stimuli, stimulus_duration,
                          set_size, set_transform_function, set_transform_params,
                          shuffle_set_order=True) -> tuple:
    """
    This function takes two original_stimuli of identical stimulus_duration,
    and applies the set_transform_function with set_transform_params to each
    stimulus, collecting the transformed versions of both stimuli into the stimuli_set

    :param original_stimuli: A tuple containing two stimuli, each consisting of a collections of neurons and their spike-times
    :param set_size: number of stimuli in the final set, an equal number is generated from each original stimulus
    :param stimulus_duration: Maximal duration of the stimulus, units: ms
    :param set_transform_function: A transformation from generation.transform to be applied to the original stimuli
    :param set_transform_params: The parameters to be used with set_transformation_function
    :param shuffle_set_order: Whether to return the set shuffled or ordered by original stimulus
    :return stimuli_set: A structured numpy array of the transformed stimuli
                        containing set_size elements, with each element having two fields:
                        'stimulus': contains the transformed stimulus as a numpy array of neurons and event times
                        'label': True if stimulus originated from original_stimulus_a, False if from original_stimulus_b
    """
    # Unpack original_stimuli tuple
    original_stimulus_a = original_stimuli[0]
    original_stimulus_b = original_stimuli[1]
    # Calculate distance between original stimuli
    distance_between_origins = calc_stimuli_distance(stimulus_a=original_stimulus_a,
                                                     stimulus_b=original_stimulus_b,
                                                     stimulus_duration=stimulus_duration)
    # Determine how many transformed version to create from origin_stimulus_a and origin_stimulus_b
    num_transformed = int(set_size / 2)
    set_transform_params['num_transformed'] = num_transformed  # Add to parameter dictionary

    # Create transformed versions from both stimuli
    transformed_set_from_a = set_transform_function(stimulus=original_stimulus_a, **set_transform_params)
    transformed_set_from_b = set_transform_function(stimulus=original_stimulus_b, **set_transform_params)

    # Combine, label and also shuffle (or not)
    stimuli_set = set_tools.combine_and_label(set_a=transformed_set_from_a,
                                              set_b=transformed_set_from_b,
                                              shuffle=shuffle_set_order)
    return stimuli_set, distance_between_origins
