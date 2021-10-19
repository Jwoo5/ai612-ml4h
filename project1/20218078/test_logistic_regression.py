import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# train_x = np.load('./X_train_lr_norm.npy')
train_x = np.load('X_train_logistic.npy')
train_y = np.load('y_train.npy')

# test_x = np.load('./X_test_lr_norm.npy')
test_x = np.load('X_test_logistic.npy')
test_y = np.load('y_test.npy')

model = pickle.load(open('lr_trained.pkl', 'rb'))

train_AUROC = roc_auc_score(train_y, model.decision_function(train_x))
train_AUPRC = average_precision_score(train_y, model.decision_function(train_x))
test_AUROC = roc_auc_score(test_y, model.decision_function(test_x))
test_AUPRC = average_precision_score(test_y, model.decision_function(test_x))

with open('./20218078_logistic_regression.txt', 'w') as f:
    print('20218078', file = f)
    print(f"{train_AUROC:.4f}", file = f)
    print(f"{train_AUPRC:.4f}", file = f)
    print(f"{test_AUROC:.4f}", file = f)
    print(f"{test_AUPRC:.4f}", file = f)