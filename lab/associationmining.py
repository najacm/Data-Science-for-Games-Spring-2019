

# Import the necessary libraries.
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pprint import pprint
from itertools import combinations

# Check that data and data path is present
data_folder = ""
data_filename = "shopping.json"
#assert os.path.isdir(data_folder) and os.path.exists(data_folder + data_filename), 'Data not found. Make sure to have the most recent version!'

# Load data
with open(data_folder + data_filename) as f:
    data_raw = json.load(f)

# Examine data
print('Number of observations', len(data_raw))
pprint(data_raw[0])

num_obs = len(data_raw)
bad_feature_names = ['', 'all- purpose']

data = None  ## YOUR CODE HERE

# Make a list of all the items
all_items = []
for i in range(num_obs):
    all_items += data_raw[i]['items']

# Find the unique item names
features_names = list(np.unique(all_items))

# Remove bad item names
for bad_feature in bad_feature_names:
    features_names.remove(bad_feature)

print('feature_names')
print(features_names)
print()
# Create a one-hot matrix
num_features = len(features_names)
data = np.zeros([num_obs, num_features])
for i in range(num_obs):
    for item in data_raw[i]['items']:
        if item not in bad_feature_names:
            j = features_names.index(item)
            data[i, j] = 1

print('The data looks like this now:')
print(data)


def compute_support(X, combi):
    """ Given a one-hot array X, and a tuple, combi, indicating the columns of interest
        compute_support returns the support of the elements in combi.

        The support of an item set X is the proportion of observations in the dataset where X occurs.
        $supp(X) = \frac{\text{# observations with X}}{\text{Total # transactions}}$
    """
    pass
    ## YOUR CODE HERE
    rule = X[:, combi].all(axis=1)
    support = rule.sum() / X.shape[0]
    return support


print('Support for (0,1,2):', compute_support(data, (0, 1, 2)))

support = []
for i in range(data.shape[1]):
    support.append(compute_support(data, (i,)))

plt.figure(figsize=(10, 4))
g = sns.barplot(x=features_names, y=support, color='b')
plt.xticks(rotation=90)
plt.show()


def candidate_generation(old_combinations):
    """ Input: An NxM array of item sets,
            N: number of combinations
            M: length of each item set
        Output: A generator that produces all possible combinations
    """

    ## YOUR CODE HERE
    items_types_in_previous_step = np.unique(old_combinations.flatten())

    for old_combination in old_combinations:
        max_combination = max(old_combination)
        for item in items_types_in_previous_step:
            if item > max_combination:
                res = tuple(old_combination) + (item,)
                yield res


# Test of candidate_generation
comb = np.array([[0],[1],[2],[3]])

while len(comb)>0:
    print('Itemset length:', len(comb[0]))
    print(comb)
    print()
    new_comb = []
    for c in candidate_generation(comb):
        new_comb.append(c)
    comb = np.array(new_comb)

min_support = 0.3

# Compute / re-compute support, and set it up as a dictionary
support = []
for i in range(data.shape[1]):
    support.append(compute_support(data, (i,)))
support = np.asarray(support)
support_dict = {1: support[support >= min_support]}

# Setup all the attomic items in a dict.
ary_col_idx = np.arange(data.shape[1])
itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
rows_count = float(data.shape[0])

itemset_len = 1
while itemset_len:
    pass

    ## YOUR CODE HERE
    next_itemset_len = itemset_len + 1
    combin = candidate_generation(itemset_dict[itemset_len])
    frequent_items = []
    frequent_items_support = []

    for c in combin:
        support = compute_support(data, c)

        if support >= min_support:
            frequent_items.append(c)
            frequent_items_support.append(support)

    if frequent_items:
        itemset_dict[next_itemset_len] = np.array(frequent_items)
        support_dict[next_itemset_len] = np.array(frequent_items_support)
        itemset_len = next_itemset_len
    else:
        itemset_len = 0

all_res = []
for k in sorted(itemset_dict):
    support = pd.Series(support_dict[k])
    itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]])

    res = pd.concat((support, itemsets), axis=1)
    all_res.append(res)

res_df = pd.concat(all_res)
res_df.columns = ['support', 'itemsets']
res_df = res_df.reset_index(drop=True)

# Print results in a human readable manner
for i in range(37, len(res_df)):
    print([features_names[i] for i in res_df.itemsets[i]], '\t has support {:.4f}'.format(res_df.support[i]))


def compute_confidence():
    """ The confidence for `X->Y` is the likelihood that `Y` is purchased, if `X` is purchased.
        This is the same as the conditional probability.
        $conf(X \rightarrow Y) = \frac{supp(X \cup Y)}{supp(X)}$
    """
    pass


def compute_lift():
    """
        $lift(X\rightarrow Y) = \frac{supp(X \cup Y)}{supp(X) supp(Y)}$
    """
    pass


def compute_conviction():
    """ How much better than chance is this association?

        $conv(X \rightarrow Y) = \frac{1-supp(Y)}{1-conf(X->Y)}$
    """
    pass


## YOUR CODE HERE
def compute_confidence(supAC, supA):
    """ The confidence for `X->Y` is the likelihood that `Y` is purchased, if `X` is purchased.
        This is the same as the conditional probability.
        $conf(X \rightarrow Y) = \frac{supp(X \cup Y)}{supp(X)}$
    """
    return supAC / supA


def compute_lift(supAC, supA, supC):
    """
        $lift(X\rightarrow Y) = \frac{supp(X \cup Y)}{supp(X) supp(Y)}$
    """
    conf = compute_confidence(supAC, supA)
    return conf / supC


def compute_conviction(supAC, supA, supC):
    """ How much better than chance is this association?

        $conv(X \rightarrow Y) = \frac{1-supp(Y)}{1-conf(X->Y)}$
    """

    conf = compute_confidence(supAC, supA)
    conviction = np.empty(conf.shape, dtype=float)

    #     if not len(conviction.shape):
    #         conviction = conviction[np.newaxis]
    #         confidence = confidence[np.newaxis]
    #         sAC = sAC[np.newaxis]
    #         sA = sA[np.newaxis]
    #         sC = sC[np.newaxis]

    conviction[:] = np.inf
    conviction[conf < 1.] = ((1. - supAC[conf < 1.]) /
                             (1. - conf[conf < 1.]))

    return conviction


score_function = lambda supAC, supA, supC: compute_confidence(supAC, supA)
min_threshold = 0.3

## YOUR CODE HERE
keys = res_df['itemsets'].values
values = res_df['support'].values

frozenset_vect = np.vectorize(lambda x: frozenset(x))
frequent_items_dict = dict(zip(frozenset_vect(keys), values))

# prepare buckets to collect frequent rules
rule_antecedents = []
rule_consequents = []
rule_supports = []

# iterate over all frequent itemsets
for k in frequent_items_dict.keys():
    supAC = frequent_items_dict[k]

    # find all possible combinations
    for idx in range(len(k)-1, 0, -1):
        # of antecedent and consequent
        for c in combinations(k, r=idx):
            antecedent = frozenset(c)
            consequent = k.difference(antecedent)

            supA = frequent_items_dict[antecedent]
            supC = frequent_items_dict[consequent]
            score = score_function(supAC, supA, supC)
            if score >= min_threshold:
                rule_antecedents.append(antecedent)
                rule_consequents.append(consequent)
                rule_supports.append([supAC, supA, supC])

# generate metrics
rule_supports = np.array(rule_supports).T.astype(float)
df_res = pd.DataFrame(
    data=list(zip(rule_antecedents, rule_consequents)),
    columns=["antecedents", "consequents"])

supAC = rule_supports[0]
supA = rule_supports[1]
supC = rule_supports[2]

df_res['total_support'] = supAC
df_res['antecedent_support'] = supA
df_res['consequent_support'] = supC
df_res['confidence'] = compute_confidence(supAC, supA)
df_res['lift'] = compute_lift(supAC, supA, supC)
df_res['conviction'] = compute_conviction(supAC, supA, supC)

print(df_res)