import pandas as pd
import numpy as np
import attr
from classification import Tempotron, evaluate, batch_train
from generation import make_set_from_specs
from types import FunctionType
import os
from tools import check_folder, save_obj, load_obj, sec_to_time, gen_datestr
from time import time
import logging

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
    :param random_seed


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

    # Setting random seed for current experiment
    random_seed = attr.ib(default=np.random.randint(100000))

    stimuli_sets = attr.ib(default=dict(training=[], test=[]))

    results = attr.ib(init=False)

    rep_times = attr.ib(default=[])
    last_run_time = attr.ib(init=False)

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
        df['release_duration'] = self.set_transform_params['release_duration']
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
        self._experiment_start_time = time()

        # Set random seed
        np.random.seed(self.random_seed)

    # Set up an ad-hoc tempotron model if one was not provided

    def _single_run(self):
        # Reset model weights
        self.model.reset()
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
        if not (number_of_repetitions):
            number_of_repetitions = self.repetitions
        # Repeat experiment number_of_repetitions time
        for i in range(number_of_repetitions):
            rep_start = time()
            rep_start_date = gen_datestr()
            print(f'Running repetition #{i+1} ', end='')
            current_results = self._single_run()
            # Write from indexes returned from the single run
            for column in current_results.index:
                # Write results of current run to experimemnt results
                self.results.loc[i, column] = current_results[column]
            # Add repetition time to results
            rep_time = sec_to_time(time() - rep_start)
            self.results.loc[i, 'run_time'] = rep_time
            rep_end_date = gen_datestr()
            print(
                f' | DONE! Started at {rep_start_date}, finished at {rep_end_date}, took {rep_time}')
            self.rep_times.append(rep_time)
        self.last_run_time = sec_to_time(time() - self._experiment_start_time)

    def save(self, folder_location, experiment_name=''):
        """
        :param folder_location: Folder to save at
        :param name: name for both files
        """
        # Append underscored to experiment name if one was provided
        if experiment_name:
            if not experiment_name[-1] == '_':
                experiment_name = experiment_name + '_'

        check_folder(folder_location=folder_location)

        # --- Saving Tempotron ---
        # Determine file name for tempotron parameters
        tempotron_params_location = os.path.join(folder_location, f'{experiment_name}params.tempotron')
        # Get reference dictionary of tempotron parameters
        model_params_dict = self.model.__dict__
        # Select only the params to be saved
        model_save_params = ['number_of_neurons', 'tau', 'threshold', 'stimulus_duration', 'weights', 'eqs']
        # Create new dictionary with only desired parameters
        model_params_savedict = {key: model_params_dict[key] for key in model_save_params}

        # Handle network saving
        model_networks = model_params_dict['networks']
        # Excluding the plotting network
        if 'plot' in model_networks:
            model_networks.pop('plot')

        # Adding network sizes to parameter dictionary
        network_sizes = {name: network['number_of_stimuli'] for name, network in model_networks.items()}
        model_params_savedict['network_sizes'] = network_sizes

        # Saving tempotron parameters
        save_obj(model_params_savedict, tempotron_params_location)

        # --- Saving Experiment params and data ---
        # Determine file name for experiment parameters
        experiment_params_location = os.path.join(folder_location, f'{experiment_name}params.experiment')
        # Get reference dictionary of experiment parameters
        experiment_params_dict = self.__dict__
        # Select only the params to be saved
        experiment_save_params = ['stimuli_creation_params', 'training_params',
                                  'origin_transform_function', 'origin_transform_params',
                                  'set_transform_function', 'set_transform_params', 'repetitions',
                                  'last_run_time', 'results', 'random_seed']
        # Create new dictionary with only desired parameters
        experiment_params_savedict = {key: experiment_params_dict[key] for key in experiment_save_params}
        # Save experiment parameters
        save_obj(experiment_params_savedict, experiment_params_location)

        # Determine file name for results file
        csv_full_path = os.path.join(folder_location, f'{experiment_name}results.csv')
        # Save experiment results file
        self.results.to_csv(csv_full_path)

        @staticmethod
        def load(location):
            """
            Loads experiment file from specified location
            :param location: the full location of the experiment file to load
            """
            raise NotImplemented('Automatic loaded for experiments not yet implemented')
