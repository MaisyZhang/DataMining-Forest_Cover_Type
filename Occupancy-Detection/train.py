import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


train_csv = '/home/pzhang/DataMining/Indoor_living/train.csv'
test_csv = '/home/pzhang/DataMining/Indoor_living/test.csv'


def logistic_regression_model(X_train, y_train, X_test, y_test, features_combs_list):
    hyper_params_space = [
        {
            'penalty': ['l1', 'l2'],
            'C': [1, 1.2, 1.5],
            'random_state': [0]
        },
    ]
    for features in features_combs_list:
        print(features)
        print('===================================')
        X = X_train.loc[:, features]
        X_t = X_test.loc[:, features]

        logit = GridSearchCV(LogisticRegression(), hyper_params_space,
                             scoring='accuracy')
        logit.fit(X, y_train)

        print('Best parameters set:')
        print(logit.best_params_)
        print()

        preds = [
            (logit.predict(X), y_train, 'Train'),
            (logit.predict(X_t), y_test, 'Test1'),
        ]
        for pred in preds:
            print(pred[2] + ' Classification Report:')
            print()
            print(classification_report(pred[1], pred[0]))
            print()
            print(pred[2] + ' Confusion Matrix:')
            print(confusion_matrix(pred[1], pred[0]))
            print()
    pass

def main():

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)




    X_train = train.drop('Occupancy', axis=1)
    y_train = train['Occupancy']

    X_test = test.drop('Occupancy', axis=1)
    y_test = test['Occupancy']

    features_combs_list = [
        ('Weekend', 'WorkingHour'),
        ('Light', 'CO2'),
        ('WorkingHour', 'CO2'),
        ('CO2', 'Temperature'),
        ('Weekend', 'WorkingHour', 'Light', 'CO2'),
        ('Weekend', 'HumidityRatio'),
    ]


    # ① Logistic Regression
    logistic_regression_model(X_train, y_train, X_test, y_test, features_combs_list)

    pass


if __name__ == '__main__':
    main()
