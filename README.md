# GöwFed - A novel Federated Intrusion Detection System for IoT devices
This project intends to create a Federated Learning based Intrusion Detection System to detect malicious network traffic in IoT devices. The implementation simulates FedAVG using **TensorFlow Federated** running on Python 3.9. More specifically, **SimpleFederatedAveraging** implementation is re-adapted to work with **Gower Distance** matrices as input for the models. Although the used dataset is [TON_IOT](https://research.unsw.edu.au/projects/toniot-datasets), designed system is supposed to work with other anomaly detection datasets equally. 

Three Federated (FL) systems have been created; a vanilla version, an Autoencoder (AE) version and a version counting with an Attention Mechanism (AM). Moreover, two centralized (CNL) analogous versions are created as well, vanilla and AE; to be used as comparison baseline.

## Directory structure
- **datasets**: TON_IOT network dataset directory
- **source (main development):** source files, initialization script
- **source/init (configuration files)**: CNL and FL systems configuration files
- **source/results (results directory)**: CNL and FL systems output destination
- **libs (required packages)**: federated, gower (modified) and deatf

## Dependencies
Install the required libraries present on [deps.req](deps.req).
> Note: Python3.9 is required and pip3 package installer recommended.


## Configuration files

Two type of configuration files are admitted depending on the CNL or FL variant. Each configuration file serves the matrix creation and the model learning modules.

> Note: The name must be cnl<*n*>.ini or fl<*n*>.ini

## Dataset
Download or create a custom dataset, the implementation is adapted to [TON_IOT](https://cloudstor.aarnet.edu.au/plus/s/ds5zW91vdgjEj9i?path=%2F) which should be downloaded, extracted and placed into the [dataset](dataset/) directory.

### CNL

Located in [source/init/cnl](source/init/cnl) the files contain the following structure:
- **run_name:** run name
- **print_scr:** visualize output in terminal
- **train_size:** number of train instances
- **test_size:** number of test instances
- **epochs:** max number of training rounds with early stopping (patience 2)
- **batch_size:** hyperparameter
- **learning_rate:** hyperparameter
- **balance_data:** used in module [create_matrix_cnl.py](source/create_matrix_cnl.py)
- **outliers:** *isoltion_forest*, *svm_one_class_classifction* or *whole_dataset*
- **seed:** added for replicability

### FL

Located in [source/init/fl](source/init/fl) the files contain the following structure:
- **run_name:** run name
- **total_rounds:** total number of averaging (communication) rounds
- **rounds_per_eval:** validate the model each *k* rounds.
- **train_clients_per_round:** number of agents taking part in each averaging
- **client_epochs_per_round:** number of local epochs in each node
- **batch_size:** hyperparameter
- **test_batch_size:** hyperparameter
- **server_learning_rate:** hyperparameter
- **client_learning_rate:** hyperparameter
- **num_clients:** total number of agents in the network
- **train_size:** total number of train instances summing all nodes datasets
- **test_size:** total number of test instances summing all nodes datasets
- **outliers:** *isoltion_forest*, *svm_one_class_classifction* or *whole_dataset*
- **balance_data:** used in module [create_matrix_fl.py](source/create_matrix_fl.py)
- **print_scr:** visualize output in terminal
- **seed:** added for replicability


## Execution instructions
Once again, depending on which system is being deployed, two execution variants exist.
However, if specific data mining wants to be performed previously; outlier detection, shap values... [preprocess.ipynb](source/preprocess.ipynb) should be used.

>Note: Filtered datasets should be placed into the same location of their raw analogous inside [dataset](dataset/) directory as well.

### CNL system

Before running a CNL-IDS, previous Gower Matrix elaboration step is compulsory and has to end in first place.

> Note: The configuration file corresponding to the specified run name must exist at [source/init/cnl](source/init/cnl).

#### CNL Gower Matrix elaboration
```sh 
python3 source/create_matrix_cnl.py <run_name>
```
> Note: Created matrix is stored at [source/mats/cnl](source/mats/cnl).

#### CNL vanilla version execution
```sh 
python3 source/netw_cnl.py <run_name>
```

#### CNL AE version execution
```sh 
python3 source/netw_cnl_AE.py <run_name>
```

### FL system

Before running a FL-IDS, previous Gower Matrices elaboration step is compulsory and has to end in first place.

> Note: The configuration file corresponding to the specified run name must exist at [source/init/fl](source/init/fl).

#### FL Gower Matrices elaboration
In this case the dataset is IID splitted among the selected number of agents as well as independent train/test matrices are created for each of them.

```sh 
python3 source/create_matrix_fl.py <run_name>
```
> Note: Created matrices are stored at [source/mats/fl](source/mats/fl).


#### FL vanilla version execution
```sh 
python3 source/netw_fl.py <run_name>
```

#### FL AE version execution
```sh 
python3 source/netw_fl_AE.py <run_name>
```

#### FL AM version execution
```sh 
python3 source/netw_fl_AM.py <run_name>
```
> Note: The attention percentage is specified in the source code under the fixed constant *BEST_PERC*. 

## Results

In each experiment, the *.h5* model and the train and validation losses are saved as well as the accuracy, precision, recall, F1-score and ROC-AUC metrics. However, in the Federated versions each entry in the results corresponds to the evaluation of the learned model on an specific agent test partition. Therefore, alongisde the mentioned stats; the client ID and its local train dataset size are stored.

For better visualizing the results, [visualize_results_cnl.ipynb](source/visualize_results_cnl.ipynb) and [visualize_results_fl.ipynb](source/visualize_results_fl.ipynb) modules have been developed. They show accuracy and pr_scores of each performed experiment as well as train and validation losses collected during the learning process. In the Federated versions, additional indicators are visualized such as dataset size per agent.

>Note: Result visualization modules are pre-configured to work with 6 experiments and plot grids of 2x3. They might be adapted for other experimental setups.

## Changes in libraries

### Gower matrix elaboration

*gower_matrix_limit_cols* and *sliced_gower_matrix_limit_cols* methods have been added to [gower_dist.py](libs/gower/gower/gower_dist.py) in order to segment the matrices in train/test subsets efficiently.

### Simple Federated Averaging
Taking the original [simple_fedavg_tff.py](libs/federated/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py) as baseline; [simple_fedavg_tff.py](libs/federated/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff_AM.py) has been implemented. On it, *run_one_round1* and *run_one_round2* methods are coded to send the temporal client weights of each client to the server and select the *k* best performing agents. Then, FedAVG is performed on the selected models of the selected subset of nodes.


## Author

* **Aitor Belenguer** 

## License

MIT
