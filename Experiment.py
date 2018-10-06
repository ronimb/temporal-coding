import pandas as pd
import numpy as np
from generation.set_classes import StimuliSet
import attr
from classification import Tempotron, evaluate, batch_train
from generation.conversion import convert_stimuli_set
from tools.set_tools import split_train_test

# %%

@attr.s
class Experiment:
    """
    Class for controlling and running experiments on a StimuliSet.

    An experiment may be set up either in of the following two ways:
        Direct model specification:     instantiating the Experiment object by specifying the "model" with which it
                                        will run  (e.g a tempotron machine learning model)
        Model parameter specification:  Instantiating the Experiment object by specifying parameters for ad-hoc
                                        model creation using the model  attribute

    training parameters for the Experiment must be supplied using the train_params attribute
    typical instantionations should look like the following:

    Using Direct model specification:
    experiment = Experiment(stimuli_set=some_StimuliSet,
                            model=some_tempotron_model,
                            train_params={batch_size, training_repetitions, learning_rate, training_set_size})

    Using Model parameters specification:
    experiment = Experiment(stimuli_set=some_StimuliSet,
                            model={tau, threshold},
                            train_params={batch_size, training_repetitions, learning_rate, training_set_size})


    """
    stimuli_set = attr.ib()  # The set of stimuli to be used for the current experiment
    model = attr.ib()  # Machine learning model to use
    train_params = attr.ib()  # The parameters used for training the model (e.g batch_size, repetitions and learning_rate and training_set_size)

    results = attr.ib(default=pd.DataFrame(
        columns=('pre' ,'post', 'diff')
    ))

    def __attrs_post_init__(self):
        # Handle stimuli set conversion if needed (SLOWS PERFORMANCE IF CONVERSION NEEDED)
        if not self.stimuli_set.converted:  # In this case we have a Normal stimuli set and must convert
            convert_stimuli_set(self.stimuli_set)
        # Handle direct model or model parameter specification
        if isinstance(self.model, Tempotron):
            pass
        elif isinstance(self.model, dict):
            self.model = self._setup_tempotron(model_params=self.model)

        # Excluding training set size from train params dict to fit batch_train function
        self.training_set_size = self.train_params.pop('training_set_size')

    def _setup_tempotron(self, model_params: dict):
        """
        used for instantiating a tempotron model within the context of the experiment
        """
        number_of_neurons = int(np.unique(self.stimuli_set.stimuli['index']).shape[0]
                                / len(self.stimuli_set))
        # Instantiating a model
        model = Tempotron(number_of_neurons=number_of_neurons, stimulus_duration=self.stimuli_set.stimulus_duration,
                          **model_params)
        return model

    def run(self, number_of_repetitions):
        # Repeat experiment number_of_repetitions time
        for i in range(number_of_repetitions):
            # Split stimuli_set to training and test sets
            train_set, test_set = split_train_test(stimuli_set=self.stimuli_set,
                                                   training_set_size=self.training_set_size)
            # Evaluate accuracy over entire set
            pre = evaluate(stimuli_set=self.stimuli_set,
                           tempotron=self.model)
            # Train with only the training set
            batch_train(stimuli_set=self.stimuli_set, **self.train_params)


# %%
def run_experiment(stimuli_set: StimuliSet, training_params: dict,
                   creation_params=None):
    pass
# Create pandas array to contain number_of_repetitions items with pre, post, diff and distance
# Iterate number_of_repetitions times
#   Create stimuli set
#   TODO: Change the following behaviour, wrap the object in a training function
#   Create tempotron object
#   Create classification network
#   Calculate accuracy  pre accuracy over entire set
#   Divide set into training and test_sets (use indexes to conserve memory)
#   Train with training set
#   Calculate post accuracy over test_set
#   Calculate diff
