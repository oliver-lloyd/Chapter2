import numpy as np
from decagon_rank_metrics import apk
from sklearn.metrics import roc_auc_score, average_precision_score


def metric_check(metric_func, pred_values, actual_values, expected_score, tolerance):
    score = metric_func(actual_values, pred_values)
    diff = np.abs(expected_score - score)
    tol_check = diff <= tolerance
    return tol_check


def ap50(pred_values, actual_values):
    return apk(actual_values, pred_values, 50)


#1 AUC: ROC and PRC

##1.1 Case of no association
sample_size = 100000
y_rand = np.random.choice(2, size=sample_size)
target_rand = np.random.choice(2, size=sample_size)
expected_rand = 0.5
tolerance_rand = 0.05
assert metric_check(roc_auc_score, y_rand, target_rand, expected_rand, tolerance_rand)
assert metric_check(average_precision_score, y_rand, target_rand, expected_rand, tolerance_rand)

##1.2 Case of full association
n = 100
y_full = [1 for _ in range(n)] + [0 for _ in range(n)]
target_full = [1 for _ in range(n)] + [0 for _ in range(n)]
expected_full = 1
tolerance_full = 0
assert metric_check(roc_auc_score, y_full, target_full, expected_full, tolerance_full)
assert metric_check(average_precision_score, y_full, target_full, expected_full, tolerance_full)

##1.3 Case of opposite association
y_opp = y_full
target_opp = y_opp[::-1]
expected_opp = 0
tolerance_opp = 0
assert metric_check(roc_auc_score, y_opp, target_opp, expected_opp, tolerance_opp)
min_auprc = 0.5  # AUPRC has minimum value that is function of class imbalance, see https://icml.cc/2012/papers/349.pdf
assert metric_check(average_precision_score, y_opp, target_opp, expected_opp+min_auprc, tolerance_opp)

#2 Average precision at 50

#2.1 Case of all correct
sample_size = 1000
target_all = list(range(sample_size))
y_all = target_all
expected_all = 1
tolerance_all = 0
assert metric_check(ap50, y_all, target_all, expected_all, tolerance_all)

#2.2 Case of none correct
target_none = list(range(sample_size))
y_none = [i+sample_size for i in list(target_none)]
expected_none = 0
tolerance_none = 0
assert metric_check(ap50, y_none, target_none, expected_none, tolerance_none)

#2.3 Case of front half correct vs back half correct
sample_size = 50
target_front = list(range(sample_size))
y_front = [i if i < 0.5*sample_size else i+sample_size for i in list(target_front)]
target_back = target_front
y_back = [i if i >= 0.5*sample_size else i+sample_size for i in list(target_back)]
assert ap50(target_front, y_front) > ap50(target_back, y_back)

# Finished
print('Script ran successfully, metrics validated.')