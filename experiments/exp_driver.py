from experiment_code import *

def standard_exp_driver(splits, all_datasets, new_header=True, showplot = False,
                        x_err = 0.01, y_err = 0.01, save_models = True):
    for dataset_name in splits.keys():
        print(dataset_name)
        random_seeds = splits[dataset_name]
        for random_seed in random_seeds:
            print('split: ' + str(random_seed))
            full_comparision([dataset_name], all_datasets, random_seed=random_seed,
                             xerr=x_err, yerr=y_err, save_models=save_models,
                             new_header=new_header, showplot=showplot)
            new_header = False

def regs_vs_compute_driver(splits, all_datasets, regs):
    for dataset_name in splits.keys():
        for random_seed in splits[dataset_name]:
            r_c = regs_vs_compute(dataset_name, regs[dataset_name], 10, all_datasets, random_seed=random_seed)
            with open('reg_vs_compute.csv', 'a') as f:
                for r, c in r_c:
                    f.write(dataset_name + ',' + str(r) + ',' + str(c) + ',' + '\n')

def rule_vs_compute_driver(splits, all_datasets, max_rules=10):
    for dataset_name in splits.keys():
        for random_seed in splits[dataset_name]:
            r_c = rules_vs_compute(dataset_name, max_rules, all_datasets)
            with open('rule_vs_compute.csv', 'a') as f:
                for r, c in r_c:
                    f.write(dataset_name + ',' + str(r) + ',' + str(c) + ',' + '\n')

def unrestricted_rulefit_driver(splits, all_datasets, Cs_lst, test_size=0.2):
    for dataset_name in splits.keys():
        problem_type = all_datasets[dataset_name][0]
        pos_class = all_datasets[dataset_name][-2]

        dataset_params = [problem_type, Cs_lst, test_size, pos_class]
        seeds = splits[dataset_name]

        res_df = unrestricted_models(dataset_name, seeds, dataset_params)
        res_df.to_csv('rulefit_unrestricted.csv', mode='a')

if __name__ == "__main__":
    test_size = 0.2
    Cs_lst = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    regs = [2, 5, 10, 20, 50, 100, 200, 500, 1000]

    all_datasets = get_all_datasets()
    splits = get_splits()

    standard_exp_driver(splits, all_datasets)
    regs_vs_compute_driver(splits, all_datasets, regs)
    rule_vs_compute_driver(splits, all_datasets)
    unrestricted_rulefit_driver(splits, all_datasets, Cs_lst, test_size)


    #TODO:
    # README
    # Doctests/Documentation
    # requirements.txt
    # refactoring with elementary runs (mid priority)