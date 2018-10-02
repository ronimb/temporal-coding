from tools import set_tools
def make_stimuli_set(transform_function, transform_params,
                     frequency, number_of_neurons, stimulus_duration,
                     original_stimuli=None):
    """
    This function creates a collection of stimuli in one of the following ways:

        ---From Original stimuli---
        Two original stimuli may be specified as a tuple named "original_stimuli",
        if these are supplied, transform_function will be applied with transform_params to each
        of these stimuli and the transformed versions from each will be collected into the set.
        In this case, all other parameters except stimulus_duration (required for transformation)

        ---De novo generation---
        If no original stimuli are specified

    """
    pass