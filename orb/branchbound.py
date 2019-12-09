from orb.rules import *

def branch_and_bound(propositions, rule = ([], []), rules = [[True]]):
    exts = rule_extensions(rule, propositions)
    for ext in exts:
        augmented_rule = (rule[0] + [ext[0]], rule[1] + [ext[1]])
        rules.append(augmented_rule[0])
        branch_and_bound(propositions, augmented_rule, rules)
    return rules
        

def rule_extensions(rule, propositions_dict):
    extensions = []
    
    if not rule:
        for prop_type in propositions_dict:
            propositions = propositions_dict[prop_type]            
            for proposition in propositions:
                extensions.append((proposition, prop_type))
    else:
        valid_extension_types = []
        for prop_type in propositions_dict:
            if prop_type not in rule[1]:
                valid_extension_types.append(prop_type)
                
        for prop_type in propositions_dict:
            if prop_type in valid_extension_types:
                propositions = propositions_dict[prop_type]
                for proposition in propositions:
                    extensions.append((proposition, prop_type))
            
                    
    return extensions

def max_prop(rule):
    return len(rule)-1,rule[-1]

#def branch_and_bound(propositions, rule = [], rules = [[True]]):
#    exts = rule_extensions(rule, propositions)
#    for ext in exts:
#        augmented_rule = rule + [ext]
#        rules.append(augmented_rule)
#        branch_and_bound(propositions, augmented_rule, rules)
#    return rules
#        
#
#def rule_extensions(rule, propositions):
#    if not rule:
#        extensions = propositions
#    else:
#        index, max_ = max_prop(rule)
#        ext_max_index = propositions.index(max_)
#        extensions = propositions[ext_max_index+1:]
#    return extensions
#
#def max_prop(rule):
#    return len(rule)-1,rule[-1]


if __name__ == "__main__":
    print(1)