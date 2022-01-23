import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "data_gtzan_try_4classes.json"
RESULT_PATH = "result_neuralnet.json"

GENRE_LIST = ["bollypop",
        "carnatic",
        "ghazal",
        "sufi"]

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y


if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    data = {
        "train score": [],
        "cv score": [],
        "test score": [],
    }

    X_flattened = X.reshape(X.shape[0], -1)
    print(X_flattened.shape)
    scaler = sklearn.preprocessing.StandardScaler()
    X_flattened = scaler.fit_transform(X_flattened)
    pca = PCA(0.9)
    pca.fit(X_flattened)
    x_pca = pca.transform(X_flattened)
    print(x_pca.shape)
    X = x_pca

    # create train/test split




    model = keras.Sequential([

        # input layer
        #keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
        keras.layers.Flatten(input_shape=(1, X.shape[1])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    X = X.reshape(400,1,58)
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X.shape)


    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=400)

    y_pred = model.predict(X_test)
    classes_x = np.argmax(y_pred, axis=1)
    print (classes_x.shape)
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=classes_x).numpy()
    print (con_mat)

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=GENRE_LIST,
                              columns=GENRE_LIST)
    print (con_mat_df)
    figure = plt.figure(figsize=(5, 5))
    plt.gcf().set_size_inches(3, 3)
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    plt.savefig("cm_nn.png", bbox_inches="tight")

    with open(RESULT_PATH, "w") as fp:
        json.dump(data, fp, indent=4)

    #joblib.dump(model, 'C:\\Users\\dcsri\\PycharmProjects\\IndianMusicGithub\\nn_final.rar')

