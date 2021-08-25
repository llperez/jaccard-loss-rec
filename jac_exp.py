import numpy as np
from tqdm import tqdm
import jac_model
import jac_data
from sklearn import metrics

num_trials = 25

jac_f1 = []
bce_f1 = []
jac_auc = []
bce_auc = []
jac_brier = []
bce_brier = []
jac_jac = []
bce_jac = []

X = jac_data.get_data()

for i in tqdm(range(num_trials)):
    
    ## get a split
    X_train, X_test, X_train_scaled, X_test_scaled = jac_data.split_dataset(X)

    ## train the model on Jaccard loss and binary_crossentropy
    model_jac = jac_model.get_model(X_train, X_train_scaled, jac_model.jaccard_loss)
    model_bce = jac_model.get_model(X_train, X_train_scaled, 'binary_crossentropy')

    ## test the model
    pred_jac = model_jac.predict(X_test_scaled)
    pred_bce = model_bce.predict(X_test_scaled)


    ## compute metrics
    jac_jac.append(metrics.jaccard_score(X_test, pred_jac > 0.5, average='weighted'))
    bce_jac.append(metrics.jaccard_score(X_test, pred_bce > 0.5, average='weighted'))

    jac_brier.append(metrics.brier_score_loss(X_test.flatten(), pred_jac.flatten()))
    bce_brier.append(metrics.brier_score_loss(X_test.flatten(), pred_bce.flatten()))

    jac_f1.append(metrics.f1_score(X_test, pred_jac > 0.5, average='weighted'))
    bce_f1.append(metrics.f1_score(X_test, pred_bce > 0.5, average='weighted'))

    jac_auc.append(metrics.roc_auc_score(X_test, pred_jac))
    bce_auc.append(metrics.roc_auc_score(X_test, pred_bce))
    
    
print('F1:')
print(np.mean(jac_f1))
print(np.mean(bce_f1))

print('AUC:')
print(np.mean(jac_auc))
print(np.mean(bce_auc))

print('BRIER:')
print(np.mean(jac_brier))
print(np.mean(bce_brier))

print('JACCARD:')
print(np.mean(jac_jac))
print(np.mean(bce_jac))
