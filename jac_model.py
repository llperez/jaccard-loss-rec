import keras
from keras import layers
import keras.backend as K

## Jaccard loss function
def jaccard_loss(y_true, y_pred, smoothing=10):
    y_pred = K.tf.cast(y_pred, float)
    y_true = K.tf.cast(y_true, float)
    v1 = K.sum(y_true**2, axis=-1)
    v2 = K.sum(y_pred**2, axis=-1)
    v3 = K.sum(y_true * y_pred, axis=-1)
    
    return K.log(v1 + v2 - v3 + smoothing) - K.log(v3 + smoothing)


## Builds the model
def get_model(X_train, X_train_scaled, loss_func, enc_layer_size=40):
    input_layer = keras.Input(shape=(X_train.shape[1],))
    enc_layer = layers.Dense(enc_layer_size, activation='tanh')(input_layer)
    dec_layer = layers.Dense(enc_layer_size, activation='tanh')(enc_layer)
    output_layer = layers.Dense(X_train.shape[1], activation='sigmoid')(dec_layer)
    model = keras.Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss=loss_func)
    model.fit(X_train_scaled, X_train, epochs=500, batch_size=128, shuffle=True)    
    return model

