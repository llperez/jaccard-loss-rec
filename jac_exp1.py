import numpy as np
from tqdm import tqdm
import jac_model
import jac_data

num_trials = 1

jac_sses = []
bce_sses = []

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

    jac_pred_sse = np.sum((pred_jac - X_test)**2)
    bce_pred_sse = np.sum((pred_bce - X_test)**2)

    jac_sses.append(jac_pred_sse)
    bce_sses.append(bce_pred_sse)

print(jac_sses)
print(bce_sses)

print('-----')

print(np.mean(jac_sses))
print(np.mean(bce_sses))
