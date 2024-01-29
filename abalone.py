import requests

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

training_set = pd.read_csv("abalone_dataset.csv")
testing_set = pd.read_csv("abalone_app.csv")

sex_mapping = {'I': 0, 'F': -1, 'M': 1}

training_set['sex'] = training_set['sex'].map(sex_mapping)
testing_set['sex'] = testing_set['sex'].map(sex_mapping)

scaler = StandardScaler()

X_train = training_set.drop('type', axis=1)
y_train = training_set['type']

X_test = testing_set

combined_data = pd.concat([X_train, X_test])
scaler.fit(combined_data)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

training_model = SVC(kernel='rbf',gamma=0.1, C=10)

def cross_validate():
    loo = LeaveOneOut()
    accuracies = []
    
    confusion_mtx = [[0,0,0],[0,0,0],[0,0,0]]    

    total_folds = loo.get_n_splits(X_train_scaled)

    for fold_num, (train_index, test_index) in enumerate(loo.split(X_train_scaled), 1):
        X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        training_model.fit(X_train_fold, y_train_fold)

        y_pred_fold = training_model.predict(X_test_fold)

        accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
        accuracies.append(accuracy_fold)

        print(f"Fold {fold_num}/{total_folds} - Accuracy: {accuracy_fold:.2%}")
        
        y_test_fold_int = int(y_test_fold.iloc[0])
        y_pred_fold_int = int(y_pred_fold[0])

        print(y_test_fold_int)
        print(y_pred_fold_int)
        
        confusion_mtx[y_test_fold_int - 1][y_pred_fold_int - 1] += 1;

    overall_accuracy = sum(accuracies) / len(accuracies)
    print("Overall Accuracy: {:.2%}".format(overall_accuracy))
    
    print("matrix de confusão")
    print(confusion_mtx)
    
    for i in range(0, 3):
        print(f"Recall of type: {i + 1} = {confusion_mtx[i][i] / sum(confusion_mtx[i])}")
        print(f"Precision of type: {i + 1} = {confusion_mtx[i][i] / sum(confusion_mtx[j][i] for j in range(3))}")


def submit_answer(): 
    training_model.fit(X_train_scaled, y_train)

    y_pred = training_model.predict(X_test_scaled)

    print("Predictions on the test set:")
    print(y_pred)
    URL = "https://aydanomachado.com/mlclass/03_Validation.php"

    DEV_KEY = "jaof"

    # json para ser enviado para o servidor
    data = {'dev_key':DEV_KEY,
            'predictions':pd.Series(y_pred).to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url = URL, data = data)

    # Extraindo e imprimindo o texto da resposta
    print(" - Resposta do servidor:\n", r.text, "\n")
    
    
cross_validate()
#submit_answer()