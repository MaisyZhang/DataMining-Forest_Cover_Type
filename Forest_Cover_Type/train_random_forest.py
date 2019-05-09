import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    df = pd.read_csv('/home/pzhang/DataMining/Surface_vegetation/covtype_categorical_small.csv')
    labels = df['Cover_Type']
    df = df.drop(['Cover_Type'], axis=1)
    data = df.drop(df.columns[[10, 11]], axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit(data).transform(data)
    X = pd.DataFrame(scaled)
    X.columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                 'Horizontal_Distance_To_Fire_Points']
    X['Wilderness_Area'] = df['Wilderness_Area']
    X['Soil_Type'] = df['Soil_Type']

    # Drop Features
    X = X.drop(['Aspect', 'Slope', 'Wilderness_Area'], axis=1)

    return X, labels


def classification(X_train, X_test, Y_train, Y_test):
    clf = RandomForestClassifier().fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    acc = accuracy_score(Y_test, predictions)
    print('Accuracy: ', acc)
    report = classification_report(Y_test, predictions)
    print('\nReport: \n', report)
    pass


def main():
    X, labels = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    classification(X_train, X_test, Y_train, Y_test)
    pass


if __name__ == '__main__':
    main()
