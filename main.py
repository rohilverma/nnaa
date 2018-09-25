import numpy as np
from keras.datasets import cifar10
import keras
import pandas as pd

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_masks = [y_train.ravel() != i for i in range(10)]
test_masks = [y_test.ravel() != i for i in range(10)]

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_test = keras.utils.to_categorical(y_test, 10)

from models import get_simple_deep_cnn

for i,(train_mask, test_mask) in enumerate(zip(train_masks, test_masks)):
    y_train_temp = y_train.copy()
    y_train_temp[y_train_temp.ravel() == 9] = [i]
    y_train_temp = keras.utils.to_categorical(y_train_temp, 9)

    model = get_simple_deep_cnn               ()
    model.fit(x_train[train_mask], y_train_temp[train_mask], batch_size=32, epochs=5)

    layer_name = model.layers[-2].name
    intermediate_layer_model = keras.models.Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    in_sample_output = intermediate_layer_model.predict(x_test[test_mask])
    out_sample_output = intermediate_layer_model.predict(x_test[~test_mask])

    with open("out.txt", "a") as f:
        f.write("Iteration: "+str(i)+"\n")
        for j in range(in_sample_output.shape[1]):
            in_col = in_sample_output[:,j]
            out_col = out_sample_output[:,j]
            f.write(" ".join(("neuron", str(j), str(np.mean(in_col)), str(np.var(in_col)), str(np.mean(out_col)), str(np.var(out_col)), "\n")))


