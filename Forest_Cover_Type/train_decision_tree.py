import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import graphviz


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


def classification(X, X_train, Y_train, X_test, Y_test, dp):
    tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=dp).fit(X_train, Y_train)
    tree_data = tree.export_graphviz(tree_model, out_file=None,
                                     feature_names=list(X),
                                     filled=True, rounded=True,
                                     special_characters=True)
    graph = graphviz.Source(tree_data)
    predictions = tree_model.predict(X_test)
    acc = accuracy_score(Y_test, predictions)
    print('Accuracy: ', acc)
    report = classification_report(Y_test, predictions)
    print('\nReport: \n', report)
    pass


def main():
    # 需要调整属性

    X, labels = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
    dp_list = [5, 10, 15, 20, 30]
    for dp in dp_list:
        print('process the dp is {} !!!!!!'.format(dp))
        classification(X, X_train, Y_train, X_test, Y_test, dp)

    pass


if __name__ == '__main__':
    main()