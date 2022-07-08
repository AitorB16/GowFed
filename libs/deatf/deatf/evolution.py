"""
Here there can be found the classes responsible of the evolution of the netwroks.
This is a key part of the library, in charge of initializing and evolving the 
desired networks, using other class to achieve it. Is also important because is 
the one that the user will make use of.

:class:`~Evolving` class is the main class and the one in charge of evolving model descriptors.
Those descriptors and the atributes specified in the initialization will generate
network to evolve; so the stepts to follow are:
        
    1. Initialize this class with all the desired qualities.
    2. Call :func:`evolve` function.
    
:class:`~DescriptorContainer` is just an auxiliar class to help Evolving. It is 
used with the DEAP library and it represent the individul to be evolved. 

================================================================================================
"""

import tensorflow as tf
import numpy as np
import random

from deap import algorithms, base, creator, tools

from deatf.network import MLP, MLPDescriptor, TCNN, CNN, RNN
from deatf.mutation import MLP_Mutation, CNN_Mutation, TCNN_Mutation, RNN_Mutation
import os

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

from contextlib import redirect_stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

descs = {"CNNDescriptor": CNN, "MLPDescriptor": MLP, "TCNNDescriptor": TCNN, "RNNDescriptor": RNN}

best_found = 100000.0


class DescriptorContainer(object):
    """
    Auxiliar class for DEAP algorithm. This object will represent the individual
    that will be evolved in the Genetic Algorithm with DEAP. It contains a dictionary
    with the networks descriptors and the hyperparameters that will be evolved.
    In this dictionary 'nX' will be the key for each network descriptor (where X is the
    number of the descripor, starting from 0) and 'hypers' will be the key for the 
    hyperparameters.
    This class does not require a fitness attribute because it will be added 
    later by the DEAP's creator. 
    
    :param desc_list: A dictionary containing the descriptors of the 
                            networks and hyperparameters that will be evolved 
                            with DEAP algorithm.
    """
    def __init__(self, desc_list):
        self.desc_list = desc_list


class Evolving:
    """
    This is the main class and contains all the needed functions and atributes
    in order to realize the evolution. As is can be seen this class has many 
    atributes to initialice, this is due to its high cutomization. Many of 
    the parameters (like evol_kwargs or sel_kwargs) are not neccesary unless 
    custom evoluationary or selection functions are used. 
    
    In order to facilitate the use of this class, here is a table with the atributes
    used in the initialization of it. Is divided in three columns:
        
    * Required atributes: atributes that always have to be declared, because
      otherwise the initialization will be inclompleted and it will give an error.
    * Predefined atributes: atributes that are necesary and have to be declared but
      they already predefined with a value and there is no need to define them 
      (it will run, but with the predefined values), but they can be defined by the 
      user and custom the execution. 
      For example, evol_alg is intialize with 'mu_plus_lambda' algorithm and if Evolving 
      class is created without asigning a value to 'evol_alg it will run with it; but it
      can be declared and defined with 'mu_comm_lambda' or 'simple' algorithms.
    * Optional atributes: atributes that are not needed, it can be initialiced without
      defining these atributes. Even so, they can be defined and the initialization will
      be more custom and with more options.
    
    =======================  =========================  ============================
    Required attributes      Predefined attributes      Optional attributes
    =======================  =========================  ============================
    n_inputs                 evol_alg                   evol_kwargs
    n_outputs                sel                        sel_kwargs
    desc_list                max_num_neurons            seed
    x_trains                 max_num_layers             hyperpatameters 
    y_trains                 max_filter                 custom_mutations
    x_tests                  max_stride
    y_tests                  dropout
    population               batch_norm
    generations              lrate
    iters                    cxp 
    batch_size               mtp
    evaluation               add_obj
                             compl
    =======================  =========================  ============================
    
    By using the described atributes in the initialization and some functions defined
    in this class, an Evolving object is created. This is the first step in the evolving
    process and the most relevant one, because here is where decisions are taken. Then 
    by calling :func:`evolve` function everything is done automaticaly.
    
    :param n_inputs: List of lists with the dimensions of the input sizes of each network.
    :param n_outputs: List of lists with the dimensions of the output sizes of each network.
    :param max_num_layers: Maximum number of layers.
    :param max_num_neurons: Maximum number of neurons in all layers (only relevant with MLP descriptors).
    :param max_filter: Maximum size of filter (only relevant with CNN and TCNN descriptors).
    :param max_stride: Maximum size of stride (only relevant with CNN and TCNN descriptors).
    :param descriptors: List with all the network descriptors that are wanted to be evolved.
    :param evaluation: Function for evaluating the model. A string in simple cases ('MSE' or 'XEntropy')
                       and one defined by the user in complex cases.
    :param batch_size: Number of samples per batch are used during training process.
    :param lrate: Learning rate used during training process.
    :param iters: Number of iterations that each model is trained.
    :param train_inputs: Dictionary with the trainig input features for each network. The key is 'iX'
                         being X the network for which the data is and the value is the actual data.
    :param train_outputs: Dictionary with the trainig output labels for each network. The key is 'oX'
                          being X the network for which the data is and the value is the actual data.
    :param test_inputs: Dictionary with the testing input features for each network. The key is 'iX'
                        being X the network for which the data is and the value is the actual data.
    :param test_outputs: Dictionary with the testing output labels for each network. The key is 'oX'
                         being X the network for which the data is and the value is the actual data.
    :param complex: A boolean valeu that indicates if the network that is going to be evaluated is complex
                    or not. The conditions for being complex are: if is indicated in initialization, if 
                    user defined evalutaion method is used, if more than one network descriptor are 
                    evolved or if the network descriptor to be evolved is different from a MLP. It is complex
                    if occurs any of these conditions; otherwise, if none of the condition are true,
                    is is considered simple (not complex).
    :param toolbox: Object Toolbox() from deap.base in the DEAP algorithm.
    :param selection: String that indicates the selection method used in the evolution. 
                      Possibilities are: 'best', 'tournament', 'roulette', 'random' or 'nsga2'.
    :param evol_alg: String that indicates the evolution algorithm that will be used.
                   Possibilities are: 'simple', 'mu_plus_lambda' or 'mu_comm_lambda'.
    :param evol_kwargs: Dictionary with parameters for the main DEAP function. The keys 
                        for that parameters are:' mu', 'lambda', 'cxpb' or 'mutpb'. Being
                        mu and lambda to float values for the mu_plus_lambda algorithm 
                        from DEAP. And cxpb and mutpb the crossover and mutation probabliites
                        respectively.
    :param cXp: Float value indicating the crossover probability. It will be 0 if there is only
                one descriptor to be evolved.
    :param mtp: Float value indicating the crossover probability. It will be 1 if there is only
                one descriptor to be evolved.
    :param generations: Number of generations that the evolution algorithm will be running.
    :param population_size: Number of individuals that will be evaluated in each generation 
                            of the evolution algorithm.
    :param hypers: Hyperparameters to be evolved in the algorithm (e.g., optimizer, batch size).
    """

    def __init__(self, desc_list, x_trains, y_trains, x_tests, y_tests, 
                 evaluation, n_inputs, n_outputs, batch_size, population, generations, iters, 
                 lrate=0.01, sel='best', max_num_layers=10, max_num_neurons=100, 
                 max_filter=4, max_stride=3, seed=None , cxp=0, mtp=1, compl=False, 
                 dropout=False, batch_norm=False, run_name='default',evol_kwargs={}, sel_kwargs={}, 
                 evol_alg='mu_plus_lambda', hyperparameters={}, custom_mutations={}, add_obj=0):

        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        self.run_name = run_name

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.max_num_layers = max_num_layers    
        self.max_num_neurons = max_num_neurons
        self.max_filter = max_filter
        self.max_stride = max_stride
        self.desc_list = desc_list
        self.evaluation = None   
                                       
        # Define and assign evalutaion function (both if is simple or complex case).
        self.define_evaluation(evaluation)

        self.batch_size = batch_size
        self.lrate = lrate
        self.iters = iters
        self.train_inputs = {}                                          
        self.train_outputs = {}
        self.test_inputs = {}
        self.test_outputs = {}
        
        # Load data in the previous dicts.
        self.data_save(x_trains, y_trains, x_tests, y_tests)            
        
        self.complex = self.is_complex(compl, evaluation, hyperparameters)
        
        self.toolbox = base.Toolbox()

        self.selection = None
        self.define_selection(sel)

        self.evol_alg = None
        self.evol_kwargs = evol_kwargs                                  
        self.cXp = cxp if len(desc_list) > 1 else 0
        self.mtp = mtp if len(desc_list) > 1 else 1

        self.generations = generations
        self.population_size = population
        self.hypers = hyperparameters

        self.define_evolving(evol_alg)
        
        # Initialize DEAP-related matters.
        self.initialize_deap(sel, sel_kwargs, batch_norm, 
                             dropout, custom_mutations, add_obj)     
        
    def data_save(self, x_trains, y_trains, x_tests, y_tests):
        """
        Load data given by parameters in the atributes of data from the class Evolving, 
        it initialices those atributes (train_inputs, train_outputs, test_inputs and 
        test_outputs). That data can be given in dictionary or list format, if is 
        given as dictionary inputs' key must be 'iX' and outputs' 'oX' (being X the 
        number of data it is, first data will hava 'i0' and 'o0').
        
        :param x_trains: Features data for training.
        :param y_trains: Labels data for training.
        :param x_tests: Features data for testing.
        :param y_tests: Labels data for testing.
        """
        if isinstance(x_trains, dict):
            self.train_inputs = x_trains
            self.train_outputs = y_trains

            self.test_inputs = x_tests
            self.test_outputs = y_tests
        else:
            for i, x in enumerate(x_trains):
                self.train_inputs["i" + str(i)] = x
                self.train_outputs["o" + str(i)] = y_trains[i]

            for i, x in enumerate(x_tests):
                self.test_inputs["i" + str(i)] = x
                self.test_outputs["o" + str(i)] = y_tests[i]

    def define_evaluation(self, evaluation):
        """
        Define the evaluation function. It accepts an string ('MSE' or 'XEntropy)
        and it will use the predifined functions from TensorFlow; this will be done
        in simple cases. Otherwise, the evaluation function will be defined and passed
        by parameter by the user. If is defined by the user it has to folloe the next
        structure:
            
        * It must have the next parameters in the following order:
            * nets (dictionary with the networks descriptors, where key 'n0' has the first 
              network descriptor).
            * train_inputs (data for the training).
            * train_outputs (expected outputs for the training).
            * batch_size (size of the batch that is going to be taken from the train data).
            * iters (number of iterations that each network will be trained).
            * test_inputs (data for testing).
            * test_outputs (expected outputs for the testing).
            * hyperparameters (dictionary with the hyperparameters like 'optimizer' or 'lrate'
              that also being evolved).
        * The output must be: value, . It must be like that because it has to receive more than
          one output. The value is the fitness or evaluation value calculated in the function.
        
        :param evaluation: Evaluation function. Either string (predefined) or customized by the user.
        """
        evals = {"MSE": tf.losses.mean_squared_error, "XEntropy": tf.nn.softmax_cross_entropy_with_logits}

        if type(evaluation) is str:
            self.evaluation = evals[evaluation]
        else:
            self.evaluation = evaluation
    
    def define_selection(self, selection):
        """
        Define the selection method for the evolution algorithm. It can be used a predefined method
        from DEAP library ('best', 'tournament', 'roulette', 'random' and 'nsga2') or a selection method 
        defined by the user. If is defined by the user it has to follow the next structure:
        as parameters recieve at least individuals (list with all the individuals) and k (the number
        of individuals that will be selected) and as output it will return the list with the 
        selected individuals. Apart from the individuals and the number of selected ones, more 
        parameters can be used, thses should be defined in 'sel_kwargs' in the initialization.
        
        :param selection: Selection function, Either string that indicates the selection method 
                          ('best', 'tournament', 'roulette', 'random' or 'nsga2') or customized 
                          by the user.
        """
        sel_methods = {'best':tools.selBest, 'tournament':tools.selTournament, 'roulette':tools.selRoulette, 
                       'random':tools.selRandom, 'nsga2':tools.selNSGA2}
       
        if type(selection) is str:
            self.selection = sel_methods[selection]
        else:
            self.selection = selection
            
    def define_evolving(self, evol_alg):
        """
        Define the evolutionary algorithm for the evolution. It uses predefined algorithms
        from DEAP library ('simple', 'mu_plus_lambda' or 'mu_comm_lambda') or a custom
        function defined by the user. This custom evolutionary algorithm function must follow
        the next structure:
            
        * At least it has to receive the following parameters: population, toolbox, cxpb, 
          mutpb, ngen, stats=None, halloffame=None.
        * More parameters can be used, they will have to be defined in evol_kwargs parameter
          in the initialization.
        * It has to return population (the final population) and logbook (object from deap.tools.Logbook 
          with the statistics of the evolution).
        * During the procces of the defined function, other functions defined in the
          toolbox can be used (selection, muatation or evalutaion).
        
        :param evol_alg: Evolutionary algorithm. It can be either a string that indicates the 
                       evolutionary algorithm ('simple', 'mu_plus_lambda' or 'mu_comm_lambda')
                       or a defined evolutionary function by the user.
        """
        deap_algs = {'simple':algorithms.eaSimple, 'mu_plus_lambda': algorithms.eaMuPlusLambda, 
                     'mu_comm_lambda':algorithms.eaMuCommaLambda}
        
        if type(evol_alg) is str:
            
            self.evol_alg = deap_algs[evol_alg]
        
            if not self.evol_kwargs:
                if evol_alg == 'simple':
                    self.evol_kwargs = {"cxpb": self.cXp, "mutpb": self.mtp}
                else:
                    self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, 
                                        "cxpb": self.cXp, "mutpb": self.mtp}
        else:
            self.evol_alg = evol_alg
    
    def is_complex(self, compl, evaluation, hyperparameters):
        """
        Determines if the case that will be evolved is a simple or a complex case.
        This will affect in the evaluation method used :func:`simple_eval` or 
        :func:`complex_eval`.
        
        :param compl: A boolean value that directly indicates if is a simple or 
                      complex case.
        :param evaluation: Evaluation parameter that is used for initialization 
                           (string or function).
        :param hyperparameters: Dictionary with the hyperparameters to ve evolved.
        :return: True boolean value if is a complex case and False boolean value
                 if is a simple case.
        """
        if compl:
            return True
        elif type(evaluation) is not str:
            return True
        elif self.desc_list[0] is not MLPDescriptor:
            return True
        elif len(self.desc_list) > 1 or len(hyperparameters) > 0:
            return True
        
        return False
        
    
    def initialize_deap(self, sel, sel_kwargs, batch_norm, dropout, custom_mutations, add_obj):
        """
        Initialize DEAP function and atributes in order to be ready for evolutionary algorithm.
        In this function all the other functions that have been defined in :func:`define_evaluation`,
        :func:`define_evolving` and :func:`define_selection` will be added to the toolbox of DEAP.
        Also here the individuals :class:`~DescriptorContainer` will be assigned as individuals 
        to be evolved.
        
        :param sel: Selection method.
        :param sel_kwargs: Hyperparameters for the selection methods (e.g., size of the tournament 
                           if that method is selected).
        :param batch_norm: Whether the evolutive process includes batch normalization in the 
                           networks or not.
        :param dropout: Whether the evolutive process includes dropout in the networks or not.
        :param custom_mutations: List with the desired mutations to be applied.
        :param add_obj: Number of additional objectives.
        """

        creator.create("Fitness", base.Fitness, weights=[-1.0]*(len(self.test_outputs) + add_obj))
        
        creator.create("Individual", DescriptorContainer, fitness=creator.Fitness)

        self.toolbox.register("individual", self.init_individual, creator.Individual, batch_norm, dropout)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("mate", cross, creator.Individual)
        self.toolbox.register("mutate", mutations, self.hypers, batch_norm, dropout, custom_mutations)

        self.toolbox.register("select", self.selection, **sel_kwargs)


    def evolve(self):
        """
        Function that actualy applies the evolutionary algorithm. Using all the information
        provided in the initialization of the class, this function does the evolution. It will
        print the mean, standard, minimum and maximum values obtained form the individuals in
        each generation. Finally, it return the individuals from the last generation, the stats
        and the best individuals found during the algorithm.
        
        :return: The last generation, a log book (stats) and the hall of fame (the best 
                 individuals found).
        """

        pop = self.toolbox.population(n=self.population_size)
        hall_of = tools.HallOfFame(self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        result, log_book = self.evol_alg(pop, self.toolbox, ngen=self.generations, 
                                       **self.evol_kwargs, verbose=1, 
                                       stats=stats, halloffame=hall_of)

        return result, log_book, hall_of

    def init_individual(self, init_ind, batch_norm, dropout):
        """
        Initializes the individual that is going to be used and evolved during the evolutionary
        algorithm. That individual will be used as dictionary with the string network id as key 
        and the network descriptor as a value, i.e., {"net_id": net_desc}. In simple case there
        will only be one network that is a MLP, in complex cases more than one network can be 
        evaluated.
        
        :param init_ind: DEAP function for transforming a network descriptor, or a list of 
                         descriptors + evolvable hyperparameters into a DEAP individual.
        :param batch_norm: A boolean value that indicates whether batch normalization is 
                           included into the evolution or not.
        :param dropout: A boolean value that incidates whether dropout is included into 
                        the evolution or not.
        :return: A DEAP individual totaly initialized.
        """

        network_descriptor = {}

        if not self.complex:  # Simple case
            network_descriptor["n0"] = MLPDescriptor()
            network_descriptor["n0"].random_init(self.train_inputs["i0"].shape[1:], self.train_outputs["o0"].shape[1], 
                                                 self.max_num_layers, self.max_num_neurons, None, None, dropout, batch_norm)
        else:  # Complex case
            for i, descriptor in enumerate(self.desc_list):
                network_descriptor["n" + str(i)] = descriptor()
                network_descriptor["n" + str(i)].random_init(self.n_inputs[i], self.n_outputs[i], self.max_num_layers, 
                                                             self.max_num_neurons, self.max_stride, self.max_filter, 
                                                             dropout, batch_norm)
        network_descriptor["hypers"] = {}
        if len(self.hypers) > 0:

            for hyper in self.hypers:
                network_descriptor["hypers"][hyper] = np.random.choice(self.hypers[hyper])

        return init_ind(network_descriptor)

    def eval_individual(self, individual):
        """
        Function used for evaluating a DEAP individual during the evolutionary algorithm.
        This is the registered function for evalution and is an auxiliar function because
        it only does another calling depending on the type of evaluation (:func:`simple_eval` or 
        :func:`complex_eval`). 
        
        :param individual: DEAP individual.
        :return: Value obtained from the evaluation.
        """
        if not self.complex:
            ev = self.simple_eval(individual)
        else:
            ev = self.complex_eval(individual)
        return ev

   
    def simple_eval(self, individual):
        """
        Evaluation in the simple case. Function for evolving a single individual. 
        No need of the user providing a evaluation function.
        
        :param individual: DEAP individual
        :return: Value obtained from the evaluation.
        """

        global best_found
        
        net = MLP(individual.desc_list["n0"])
        
        inp = Input(shape=self.n_inputs[0])
        out = Flatten()(inp)
        out = net.building(out)
        model = Model(inputs=inp, outputs=out)

        opt = tf.keras.optimizers.Adam(learning_rate=self.lrate)
        model.compile(loss=self.evaluation, optimizer=opt, metrics=[])
        
        model.fit(self.train_inputs['i0'], self.train_outputs['o0'], epochs=self.iters, batch_size=self.batch_size, verbose=0) #1 show loss
        
        ev = model.evaluate(self.test_inputs['i0'], self.test_outputs['o0'], verbose=0)
        
        if(ev < best_found):
            best_found = ev
            model.save('/home/tester/Desktop/clean_repo/FL-IDS/priscilla/NNs/'+ self.run_name + '_model.h5')
            with open('/home/tester/Desktop/clean_repo/FL-IDS/priscilla/NNs/' + self.run_name + '_nn_summary.txt', 'w') as f:
                with redirect_stdout(f):
                    model.summary()
            #model.summary(print_fn=myprint)
            #model.summary()


        if isinstance(ev, float):
            ev = (ev,)
            
        return ev

    def complex_eval(self, individual):
        """
        Evaluation in the complex case. Function for evolving individuals in a 
        complex case. The user must have implemented the training and evaluation functions.
        
        :param individual: DEAP individual
        :return: Value obtained from the evaluation.
        """
        nets = {}
    
        for index, net in enumerate(individual.desc_list.keys()):
            if "hypers" not in net:
                nets[net] = descs[self.desc_list[index].__name__](individual.desc_list[net])

        ev = self.evaluation(nets, self.train_inputs, self.train_outputs, self.batch_size, self.iters,
                             self.test_inputs, self.test_outputs, individual.desc_list["hypers"])

        return ev


def mutations(hypers, batch_norm, dropout, custom_mutations, individual):
    """
    Mutation operators for individuals. It can be affected any network or hyperparameter.
    Depending on the type of network that will suffer the mutation, this function
    will create a different object from :class:`deatf.mutation.Mutation`.
    
    :param hypers: Hyperparameters not included in the networks to be evolved.
    :param batch_normalization: Whether batch normalization is part of the evolution or not.
    :param dropout: Whether dropout is part of the evolution or not.
    :param individual: DEAP individual. Contains a dict where the keys are the components of the model.
    :return: Mutated version of the DEAP individual.
    """
    mutation_types = {'MLPDescriptor': MLP_Mutation, 'CNNDescriptor': CNN_Mutation, 
                      'TCNNDescriptor': TCNN_Mutation, 'RNNDescriptor': RNN_Mutation}

    nets = list(individual.desc_list.keys())
    hyperparameters = individual.desc_list["hypers"]
    nets.remove("hypers")

    network = individual.desc_list[np.random.choice(nets)]

    if not custom_mutations:
        network_custom_mutations = [] # If no custom mutations are passed, each network's mutations will be applied
    else:
        network_custom_mutations = custom_mutations[network.__class__.__name__]
    
    network_mutation = mutation_types[network.__class__.__name__](hypers, batch_norm, dropout, network, 
                                                                  hyperparameters, network_custom_mutations)
    network_mutation.apply_random_mutation()
    
    return individual,

def cross(init_ind, ind1, ind2):
    """
    Crossover operator for individuals. Cannot be applied in the simple case, as it randomly 
    interchanges model components.
    
    :param init_ind: DEAP function for initializing dicts (in this case) as DEAP individuals.
    :param ind1: 1st individual to be crossed (first parent).
    :param ind2: 2st individual to be crossed (second parent).
    :return: Two new DEAP individuals, the crossed versions of the incoming 
             parameters (the offspring).
    """

    keys = list(ind1.desc_list.keys())
    # Randomly select the keys of the components that will be interchanged.
    cx_point = np.random.choice(keys, size=np.random.randint(1, len(keys)) if len(keys) > 2 else 1, replace=False)
    new1 = {}
    new2 = {}

    for key in keys:
        if key in cx_point:
            new1[key] = ind1.desc_list[key]
            new2[key] = ind2.desc_list[key]
        else:
            new1[key] = ind2.desc_list[key]
            new2[key] = ind1.desc_list[key]


    return init_ind(new1), init_ind(new2)
