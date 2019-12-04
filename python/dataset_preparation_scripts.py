import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_dataset_test_train_separate(dataset_name, test_name, train_name, Y_name, cols_to_drop):
    dataset_dir = os.path.join(os.getcwd(), "Datasets", dataset_name)

    train = pd.read_csv(os.path.join(dataset_dir, train_name))
    test = pd.read_csv(os.path.join(dataset_dir, test_name))

    train_df = pd.DataFrame(train)
    train_df = train_df.drop(cols_to_drop, axis='columns')
    train_df = train_df.dropna()


    test_df = pd.DataFrame(test)
    test_df = test_df.drop(cols_to_drop, axis='columns')
    test_df = test_df.dropna()
    
    X_test, Y_test = X_Y_Split(test_df, Y_name)
    X_train, Y_train = X_Y_Split(train_df, Y_name)

    data = [X_train, Y_train, X_test, Y_test]
    
    return data

def X_Y_Split(df, Y_name):
    Y = df[Y_name]
    X = df.drop([Y_name], axis='columns')
    X = pd.get_dummies(X)
    
    return X, Y


def prepare_dataset_one_file(dataset_name, file_name, Y_name, split_size, cols_to_drop):
    dataset_dir = os.path.join(os.getcwd(), "Datasets", dataset_name)

    dataset = pd.read_csv(os.path.join(dataset_dir, file_name))

    df = pd.DataFrame(dataset)
    df = df.drop(cols_to_drop, axis='columns')
    df = df.dropna()

    Y = df[Y_name]
    X = df.drop([Y_name], axis='columns')
    X = pd.get_dummies(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=split_size)
    
    data = [X_train, Y_train, X_test, Y_test]

    return data

'''
def prepare_titanic():
    titanic_dir = "\\".join(os.getcwd().split("\\")[:-2]) + "\\Datasets\\titanic"
        
    train = pd.read_csv(titanic_dir + "\\train.csv")
    train['label'] = 'train'
    test = pd.read_csv(titanic_dir + "\\test.csv")
    test['label'] = 'test'

    survival = pd.read_csv(titanic_dir + "\\gender_submission.csv")

    # Any results you write to the current directory are saved as output.

    # create training data frame
    train_df = pd.DataFrame(train)
    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis='columns')
    #train_df = train_df.apply (pd.to_numeric, errors='coerce')
    train_df = train_df.dropna()
    train_df = train_df.drop(['label'], axis='columns')
    #train_df = train_df.reset_index(drop=True)
    train_df.head()

    # creating test data fram

    test_df = pd.DataFrame(test)

    survival_df = pd.DataFrame(survival)

    test_df['Survived'] = survival_df['Survived']

    # survival_df = survival_df.drop(['PassengerId'], axis = 'columns')
    # test_df['Survived'] = survival_df

    test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis='columns')
    test_df = test_df.dropna()
    test_df = test_df.drop(['label'], axis='columns')

    # X, Y train
    Y_train = train_df['Survived']
    X_train = train_df.drop(['Survived'], axis='columns')
    X_train  = pd.get_dummies(X_train)

    #imputer = SimpleImputer()
    #X_train = imputer.fit_transform(X_train)

    X_train.head()

    # X, Y test
    Y_test = test_df['Survived']
    X_test = test_df.drop(['Survived'], axis='columns')
    X_test = pd.get_dummies(X_test)
    
    data = [X_train, Y_train, X_test, Y_test]

    return data
'''