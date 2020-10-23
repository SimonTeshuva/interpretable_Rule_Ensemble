# interpretable_Rule_Ensambles_Experiments

 Code for to perform experiments comparing the effectiveness of optimal
 rule boosting, greedy rule boosting and rulefit. Used to
  create tables of results used in the paper 
  "Better Short than Greedy: Interpretable Models through Optimal Rule Boosting"

The datasets directory includes all the datasets used to create the tables of results in the above paper. 
The experiments folder contains the python scripts used to perform the experiments on the datasets.

exp_driver.py: a collection of wrapper functions for convieniently replicating the experiments perfomred 
to generate tables of results

experiment_code.py: a collection of functions used to perform the experiments and generate tables of results

dataset_prep.py: data pre-processing performed on all datasets before being fitted

dataset_info.py: contains functions that return parameters used for each dataset

exp_setup_new.py: oontains all the code from the four above mentioned files as well as some legacy code
## Installation


```bash
pip install -r requirements.txt
```

## Usage

```python
import exp_driver

all_datasets = get_all_datasets() # gets all the information used when when creating the original tables of results.
# Includes (in order, for each dataset): problem_type (r or c), downsample_size (on dataset), max_rules,
# (grd_reg, opt_reg), max_col_attr (for rule boosting), rulefit_reg, test_size, repeats, pos_class,
# opt_max_rules (in case fewer rules were fitted for optimal)

splits = get_splits() # gets the same random splits that were originally 
# randomly generated when creating the tables of results.
# Alternatively could feed a vector of 'None' to generate a new random table



# each of the function calls below creates a .csv file with comparitive results.

standard_exp_driver(splits, all_datasets) # compare rulefit, greedy rule booting and optimal rule boosting.
# in the context of comprehensible rule ensembles
 
regs = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
regs_vs_compute_driver(splits, all_datasets, regs) # compute time vs regularisation for optimal rule boosting

rule_vs_compute_driver(splits, all_datasets) # ensemble size vs compute time for optimal rule boosting. 

test_size = 0.2
Cs_lst = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
unrestricted_rulefit_driver(splits, all_datasets, Cs_lst, test_size) # rulefit for a variety of Cs values which create
# large, not necessarily comprehensible rule ensembles. 
```

## Contributing

## License
