import json
import numpy as np
import sklearn.svm as SVM
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "data_IndianMusicAll.json"

GENRE_LIST = ["bollypop",
        "carnatic",
        "ghazal",
        "sufi"]

RESULT_PATH = "result_svm.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y

def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    data = {
        "train score": [],
        "cv score": [],
        "test score": [],
    }
    print (X.shape, y.shape)
    X_flattened = X.reshape(X.shape[0], -1 )
    print (X_flattened.shape)
    scaler = sklearn.preprocessing.StandardScaler()
    X_flattened = scaler.fit_transform(X_flattened)
    pca = PCA(0.9)
    pca.fit(X_flattened)
    x_pca = pca.transform(X_flattened)
    print(x_pca.shape)
    X_flattened = x_pca


    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.3)

    # build svm model
    model = sklearn.svm.SVC(kernel = 'rbf')
    #model = RandomForestClassifier()
    #model = sklearn.neighbors.KNeighborsClassifier()
    #base = SVM.SVC(kernel= 'linear')
    #base = DecisionTreeClassifier(max_depth = 20)

    #model = AdaBoostClassifier(base_estimator= base)
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print("Training Accuracy:{}".format(train_accuracy))
    print("Testing Accuracy:{}".format(test_accuracy))
    data["train score"] = train_accuracy
    data["test score"] = test_accuracy

    cm = confusion_matrix(y_test, model.predict(X_test))
    print ("Confusion Matrix:")
    print (confusion_matrix(y_test, model.predict(X_test)))
    print ("total num of test samples = {}".format(cm.sum()))

    scores = evaluate_model(model, X_flattened, y)
    print ("CV : Mean Score:{}, Std Dev : {}".format( np.mean(scores), np.std(scores)))
    data["cv score"] = np.mean(scores)

    con_mat = cm
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=GENRE_LIST,
                              columns=GENRE_LIST)
    print(con_mat_df)
    figure = plt.figure(figsize=(5,5))
    plt.gcf().set_size_inches(3, 3)
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("cm.png", bbox_inches="tight")

    with open(RESULT_PATH, "w") as fp:
        json.dump(data, fp, indent=4)

    joblib.dump(model, 'C:\\Users\\dcsri\\PycharmProjects\\IndianMusicGithub\\svm_indian_final.rar')