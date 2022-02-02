# robustness-albert

This is the code for ["How Emotionally Stable is ALBERT? Testing Robustness with 
Stochastic Weight Averaging on a Sentiment Analysis Task"](https://aclanthology.org/2021.eval4nlp-1.3/).

Links to the models and other results will soon be released.

## setup. 
To run the training and evaluation for this paper, please set up the environment: 
```bash 
# Create environment.
conda create -n robustness-albert python=3.7
conda activate robustness-albert

# Install packages.
python setup.py develop
pip install -r requirements.txt
```

## training.
First, create a config file (see `configs/example_config.json` for an example). 

Then, run the following:
```bash
robustness_albert/train.py -c configs/CONFIG_FILE_NAME.json
```

## linting & unit testing. 
For linting and unit testing, run the following commands: 
```bash
# Linting.
flake8 robustness-albert

# Unit testing. 
pytest -s tests/
```

## notebooks (coming soon).
`checklist_test_models.ipynb`: This notebook carries out the CheckList tests on 
all the random seeds. 

`dev_set_results.ipynb`: This notebook loads the results of the different random seeds
 on the development set of SST-2 and calculates the Fleiss' Kappa agreement between 
 the models.
 
 `extract_names_sst2.ipynb`: This notebook extracts and saves names that occur in the
 movie reviews of the train and test set, so we can use these names for the designed 
 CheckList capabilities. Resulting names can be found in `assets/names_sst2_train.json`
 and `assets/names_sst2_test.json`
 
 `plot_checklist_results.ipynb`: This notebook plots the results achieved from the 
 CheckList tests for all random seeds. It plots the error rates and overlap ratios and 
 calculates the Fleiss' Kappa agreement.
 
 `sst2_test_labels.ipynb`: As the SST-2 dataset in _HuggingFace_ does not come with the 
 test labels, this notebook is used to extract them using the [original SST-2 data from
 GLUE](https://gluebenchmark.com/tasks). 
 Labels can be found in `assets/sst2_test_labels.json`
 
 `checklist_sst2.ipynb`: This notebook creates the CheckList test suite that we use for 
 the results. Resulting test suite can be found in `assets/testset_19_07_21.pkl`.
