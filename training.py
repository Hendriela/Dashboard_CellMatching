""" Training of decision tree classifier

This script allows the user to train a decision tree classifier that predicts which neurons are the same over sessions,
by learning simple decision rules inferred from the data features. See documentation for details on the
classifier.


*** Set the list 'labelled_results_csv_paths' to the paths to the final csv files (.csv) that were saved when
using the cellmatching dashboard. These are already in the correct format. If you plan to generate csv files in another
manner check the documentation (Section 'Training the classifier') for details on the required format of the csv files.


*** Set 'train_clf_full_data':
    1. True -> train the classifier and update the resulting classifier pkl file used in the cellmatching dashboard
    2. False ->  split data into train and test sets and measure and visualize performance of the classifier on the data
    (classifier pkl file will not be updated!)

"""

import pandas as pd
from sklearn import tree
import pickle
import json
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns


# **********************************************************************************************************************
#               USER-SPECIFIC SETTINGS
# **********************************************************************************************************************

labelled_results_csv_paths = ['/home/anna/Documents/Neuron_imaging_validationsets/result_jithin_pre.csv',
                              '/home/anna/Documents/Neuron_imaging_validationsets/result_hendrik_M38.csv',
                              '/home/anna/Documents/Neuron_imaging_validationsets/result_hendrik_M41.csv']

# Set to 'True' if you want to train the classifier and update the resulting pkl file used in the dash app
# Set to 'False' to measure performance of the classifier on the data (classifier pkl file will not be updated!)
train_clf_full_data = True

# filename (or path) to save trained model, trained decision tree image, formatted training data
clf_pkl_filename = "pickle_model.pkl"
decision_tree_filename = 'decision_tree.png'
training_data_filename = 'training_data.csv'

# model settings (can leave the defaults)
always_predict_match = False    # disable the option of predicting "no match"
num_neurs_clf_input = 3         # number of closest neurons that the classifier can choose from
max_depth_decision_tree = 5
min_samples_leaf_decision_tree = 5


# **********************************************************************************************************************


def get_feature_class_names():
    """ Computes the feature and class names of the data used to train the classifier.
        (Depends on 'num_neurs_clf_input', the number of neurons that the classifier can choose from (set above))

    Returns:
        list (str): List of feature names.
        list (str): List of class names.
    """
    features = ['neuron_idx_other', 'com_x', 'com_y', 'dist', 'neur shape', 'area shape', 'angle diff',
    'num neur neighbours', 'neighbours_q1', 'neighbours_q2', 'neighbours_q3', 'neighbours_q4']
    feature_names, class_names = [], []
    for i in range(num_neurs_clf_input):
        curr_features = [feature + ' ' + str(i) for feature in features]
        feature_names.extend(curr_features)

    class_names = ['neuron ' + str(i) for i in range(num_neurs_clf_input)]
    if not always_predict_match:
        class_names.append('no match')
    return feature_names, class_names



def save_decision_tree(clf):
    """ Plots and saves the image of the decision tree of the classifier.

    Args:
        clf: Decision Tree classifier object
    """
    fn, cn = get_feature_class_names()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 15), dpi=300)
    tree.plot_tree(clf, feature_names=fn,
                   class_names=cn,
                   filled=True)
    fig.savefig(decision_tree_filename)


def measure_performance(clf, X_test, y_test_lab):
    """ Measures performance of classifier of classifier on test samples.
        Outputs accuracy, precision to the console and plots the confusion matrix.

    Args:
        clf: Decision Tree classifier object
        X_test (array-like, sparse matrix of shape (n_samples, n_features)): testing input samples
        y_test_lab (array-like of shape (n_samples)): target values
    """
    y_pred = clf.predict(X_test)
    feature_names, class_names = get_feature_class_names()

    # accuracy
    accuracy_score = metrics.accuracy_score(y_test_lab, y_pred)
    print("accuracy score: ", accuracy_score)

    # precision
    precision = metrics.precision_score(y_test_lab, y_pred, average=None)
    precision_results = pd.DataFrame(precision, index=class_names)
    precision_results.rename(columns={0: 'precision'}, inplace=True)
    print(precision_results)

    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test_lab, y_pred)
    matrix_df = pd.DataFrame(confusion_matrix)
    plt.figure(figsize=(10, 7)); ax = plt.axes()
    sns.set(font_scale=1.3); sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
    ax.set_title('Confusion Matrix - Decision Tree')
    ax.set_xlabel("Predicted label", fontsize=15)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("True Label", fontsize=15)
    ax.set_yticklabels(list(class_names), rotation=0)
    plt.show()


def save_clf(clf):
    """    Args clf: Create Decision Tree classifier object    """
    with open(clf_pkl_filename, 'wb') as file:
        pickle.dump(clf, file)


def train_clf(X, y, only_train=True):
    """ Trains the decision tree classifier.

    Args:
        X (array-like, sparse matrix of shape (n_samples, n_features)): training input samples
        y (array-like of shape (n_samples)): target values
        only_train (bool): A flag used to train on full (or partial) data.

    Returns:
        Decision Tree classifier object: trained classifier
        2D array of shape (n_samples, n_features): training input samples (empty list if training on all data)
        1D array of shape (n_samples): array representing the target values (empty list if training on all data)
    """
    X, y = shuffle(X, y)
    X_test, y_test = [], []
    if not only_train:
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = tree.DecisionTreeClassifier(max_depth=max_depth_decision_tree,
                                      min_samples_leaf=min_samples_leaf_decision_tree)
    clf = clf.fit(X, y)
    return clf, X_test, y_test

def get_y(df, feature_idx, result_idx):
    """ Compute target value for each neuron idx (of one session), as follows:
            input: (features1, features2, features3,...),
                    where featuresX is a list of the features of the X-th closest neuron

            output: 0 (1, 2, ...) if the correct neuron is the first (second, third, ...) neuron listed
                    if none of the three closest neurons was the correct neuron, the ouput is the value of
                    'num_neurs_clf_input' (set above)
        Args:
            df (pandas dataframe): holds the data of the csv files of hte labelled cell matching
            feature_idx (str): name of column of dataframe representing the features
            result_idx (str): name of column of dataframe representing the confirmed chosen neuron match

        Returns:
            2D array-like (list): Formatted features that can be used as input for the classifier
            list: Target values for the
    """
    features = json.loads(df[feature_idx])[:num_neurs_clf_input]
    matched_neuron_idx = df[result_idx]
    label = num_neurs_clf_input
    for i in range(num_neurs_clf_input):
        if matched_neuron_idx == features[i][0]:
            label = i
            break
    if always_predict_match and label == num_neurs_clf_input:
        label = 0
    features_flat_list = [item for sublist in features for item in sublist]
    return [features_flat_list, label]

def prepare_training_data(labelled_data_paths, save_training_data=False):
    """ Creates the training input samples and target values from the csv files with labelled matches.

    Args:
        labelled_data_paths (List[str]): The file locations of the csv files.
        save_training_data (bool): A flag used to save the training data to file.

    Returns:
        2D array of shape (n_samples, n_features): training input samples
        1D array of shape (n_samples): array representing the target values
    """
    X = []
    y = []
    for file_path in labelled_data_paths:
        df = pd.read_csv(file_path, sep=';')
        df = df.loc[df['confirmed'] == 1]
        feature_cols = [col for col in df if col.endswith('feature_vals')]
        for feature_col in feature_cols:
            result_col = feature_col[:-12]
            df_training_data = df.apply(get_y, feature_idx=feature_col, result_idx=result_col, axis=1)
            df_training_data = pd.DataFrame(df_training_data.to_list(), columns=['X', 'y'])
            X.extend(df_training_data['X'].to_list())
            y.extend(df_training_data['y'].to_list())
    print("total number of datapoints: ", len(X))

    if save_training_data:
        training_data = {'X': X, 'y': y}
        training_df = pd.DataFrame(training_data, columns=['X', 'y'])
        training_df.to_csv(training_data_filename, index=False, header=True)

    return X, y


def main():
    X, y = prepare_training_data(labelled_results_csv_paths, save_training_data=False)
    clf, X_test, y_test = train_clf(X, y, only_train=train_clf_full_data)
    if train_clf_full_data:
        save_clf(clf)
        save_decision_tree(clf)
    else:
        measure_performance(clf, X_test, y_test)


if __name__ == "__main__":
    main()
