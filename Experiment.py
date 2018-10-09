import pandas as pd
import numpy as np
import attr
from classification import Tempotron, evaluate, batch_train
from generation import make_set_from_specs
from types import FunctionType
import os
from tools import check_folder
import pickle
from tools import time_tools
from time import time


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

    training parameters for the Experiment must be supplied using the training_params attribute
    typical instantionations should look like the following:

    Using Direct model specification:
    experiment = Experiment(stimuli_creation_params=dict(frequency=f, number_of_neurons=n, stimulus_duration=t),
                            model=some_tempotron_model,
                            **All_other_parameters)

    Using Model parameters specification:
    experiment = Experiment(stimuli_creation_params=dict(frequency=f, number_of_neurons=n, stimulus_duration=t),
                            model={tau, threshold},
                            **All_other_parameters)

    The parameters required for running an experiment are as follows:
    :param stimuli_creation_params: Parameters for original stimulus generation
    :param model
    :param training_params
    :param origin_transform_function
    :param origin_transform_params
    :param set_transform_function
    :param set_transform_params
    :param repetitions


    After the experiment has run, the following fields will be added to it:
    :key original_stimuli
    :key original_stimuli_distance
    :key stimuli_set : The stimuli set that was generated for this experiment
    :key results


    """
    # ToDo: Generalize this to different possible experiment setups
    stimuli_creation_params = attr.ib(validator=attr.validators.instance_of(dict))
    model = attr.ib()  # Machine learning model to use
    training_params = attr.ib(validator=attr.validators.instance_of(
        dict))  # The parameters used for training the model (e.g batch_size, repetitions and learning_rate and training_set_size)
    origin_transform_function = attr.ib(validator=attr.validators.instance_of(FunctionType))
    origin_transform_params = attr.ib(validator=attr.validators.instance_of(dict))
    set_transform_function = attr.ib(validator=attr.validators.instance_of(FunctionType))
    set_transform_params = attr.ib(validator=attr.validators.instance_of(dict))
    repetitions = attr.ib(default=15)

    stimuli_sets = attr.ib(default=dict(training=[], test=[]))

    results = attr.ib(init=False)

    rep_times = attr.ib(default=[])

    def _results_initializer(self):
        data_columns = ('duration', 'frequency', 'number_of_neurons',
                        'model_threshold', 'learning_rate',
                        'orig_a_mean_freq', 'orig_b_mean_freq', 'orig_difference', 'orig_distance', 'set_size',
                        'release_duration', 'number_of_vesicles', 'release_probability',
                        'pre', 'post', 'diff')
        df = pd.DataFrame(
            np.zeros((self.repetitions, len(data_columns))),  # Currently written for a specific experiment mode
                     columns=data_columns)
        df['frequency'] = self.stimuli_creation_params['frequency']
        df['number_of_neurons'] = self.stimuli_creation_params['number_of_neurons']
        df['model_threshold'] = self.model.threshold
        df['learning_rate'] = self.training_params['learning_rate']
        df['duration'] = self.stimuli_creation_params['stimulus_duration']
        df['set_size'] = self.stimuli_creation_params['set_size']
        df['number_of_vesicles'] = self.set_transform_params['number_of_vesicles']
        df['release_probability'] = self.set_transform_params['release_probability']
        df['orig_difference'] = str(self.origin_transform_params['interval'])
        return df

    @property
    def entire_stimuli_sets(self):
        all_sets = []
        for train_set, test_set in zip(self.stimuli_sets['training'], self.stimuli_sets['test']):
            current_set = []
            current_set.extend(train_set)
            current_set.extend(test_set)
            all_sets.append(current_set)
        return all_sets

    def _setup_tempotron(self, model_params: dict):
        """
        used for instantiating a tempotron model within the context of the experiment
        """
        number_of_neurons = self.stimuli_creation_params['number_of_neurons']
        # Instantiating a model
        model = Tempotron(number_of_neurons=number_of_neurons,
                          stimulus_duration=self.stimuli_creation_params['stimulus_duration'],
                          **model_params)
        return model

    def __attrs_post_init__(self):
        # Handle direct model or model parameter specification
        if isinstance(self.model, Tempotron):
            pass
        elif isinstance(self.model, dict):
            self.model = self._setup_tempotron(model_params=self.model)
        self.results = self._results_initializer()

    # Set up an ad-hoc tempotron model if one was not provided

    def _single_run(self, stimuli_set_index=None, reassign_test_training=False):
        # Reset model weights
        self.model.reset()
        if not stimuli_set_index:
            # Generate set for this run
            stimuli_set = make_set_from_specs(
                origin_transform_function=self.origin_transform_function,
                origin_transform_params=self.origin_transform_params,
                set_transform_function=self.set_transform_function,
                set_transform_params=self.set_transform_params,
                **self.stimuli_creation_params)
            # Split into training and test
            test_set, training_set = stimuli_set.split(self.training_params['fraction_training'])

            # Add Current StimuliSet to list of StimuliSets in the Experiment object
            self.stimuli_sets['training'].append(training_set)
            self.stimuli_sets['test'].append(test_set)
        else:
            if reassign_test_training:
                # Join training and test sets together again
                stimuli_set = self.entire_stimuli_sets[stimuli_set_index]
                # Reassign test and training sets
                test_set, training_set = stimuli_set.split(self.training_params['fraction_training'])
                # Set reassigned sets
                self.stimuli_sets['training'][stimuli_set_index] = training_set
                self.stimuli_sets['test'][stimuli_set_index] = test_set

            else:
                test_set = self.stimuli_sets['test'][stimuli_set_index]
                training_set = self.stimuli_sets['training'][stimuli_set_index]



        # Handle conversion to tempotron format once
        test_set._make_tempotron_converted
        training_set._make_tempotron_converted()
        # Initial evaluation over test set
        pre = evaluate(stimuli_set=test_set,
                       tempotron=self.model)
        # Train model with training set
        batch_train(stimuli_set=training_set, tempotron=self.model, **self.training_params)
        # Evaluate post training performance
        post = evaluate(stimuli_set=test_set, tempotron=self.model)
        # Calculate difference
        diff = post - pre
        # Add more data to results
        orig_stimuli_distance = test_set.original_stimuli_distance  # It doesnt matter whether its taken from test or training as they are made from the same two stimuli
        origs_a_mean_freq = np.mean(
            np.multiply([neuron.size for neuron in stimuli_set.original_stimuli[0]],
                        1000 / self.stimuli_creation_params['stimulus_duration'])
        )
        origs_b_mean_freq = np.mean(
            np.multiply([neuron.size for neuron in stimuli_set.original_stimuli[1]],
                        1000 / self.stimuli_creation_params['stimulus_duration'])
        )
        single_results = pd.Series([origs_a_mean_freq, origs_b_mean_freq, orig_stimuli_distance, pre, post, diff],
                                   ['orig_a_mean_freq', 'orig_b_mean_freq', 'orig_distance', 'pre', 'post', 'diff'])
        return single_results

    def run(self, number_of_repetitions=None,
            reassign_test_training=False,
            regenerate_sets=False):
        """

        :param number_of_repetitions:
        :param reassign_test_training:
        :param regenerate_sets:
        :return:
        """
        if not(number_of_repetitions):
            number_of_repetitions=self.repetitions
        # Repeat experiment number_of_repetitions time
        for i in range(number_of_repetitions):
            rep_start = time()
            rep_start_date = time_tools.gen_datestr()
            print(f'Running repetition #{i+1} ', end='')
            current_results = self._single_run()
            for column in current_results.index:
                # Write results of current run to experimemnt results
                self.results.loc[i, column] = current_results[column]  # Todo: fix, this probably doesnt assign correct currently, test assignement on made up dataframes
            rep_time = time() - rep_start
            rep_end_date = time_tools.gen_datestr()
            print(f' | DONE! Started at {rep_start_date}, finished at {rep_end_date}, took {time_tools.sec_to_time(rep_time)}')
            self.rep_times.append(rep_time)

    def save(self, location, name):
        """
        :param location: Folder to save at
        :param name: name for both files
        """
        csv_fullpath = os.path.join(location, f'{name}.csv')
        experiment_fullpath = os.path.join(location, f'{name}.experiment')
        check_folder(folder_location=location)

        # Saving results csv
        self.results.to_csv(csv_fullpath)

        # Saving experiment object
        #TODO: This will have to be done in a bit of a more tricky way...
        with open(experiment_fullpath, 'wb') as experiment_file:
            pickle.dump(self, experiment_file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(location):
        """
        Loads experiment file from specified location
        :param location: the full location of the experiment file to load
        """
        with open(location, 'rb') as experiment_file:
            pickle.load(experiment_file)
