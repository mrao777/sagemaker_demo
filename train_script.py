from __future__ import print_function

import argparse
import os
import pandas as pd
import numpy as np
#from sklearn.externals import joblib
import joblib
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.impute import KNNImputer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Load Data to pandas data frame
    file = os.path.join(args.train, "campain_data.csv")
    customer_data = pd.read_csv(file)
    customer_data = customer_data.drop('id', axis=1)

    # Convert categorical features to numerical 
    customer_data['income']=customer_data['income'].map(
    {'Under $10k':1,'10-19,999':2,'20-29,999':3,'30-39,999':4,'40-49,999':5,
     '50-59,999':6,'60-69,999':7,'70-79,999':8,'80-89,999':9,'90-99,999':10,
      '100-149,999':11,'150 - 174,999':12,'175 - 199,999':13,'200 - 249,999':14,'250k+':15})
    customer_data['gender']=customer_data['gender'].map({'M':1,'F':0})
    customer_data['marital_status']=customer_data['marital_status'].map({'M':1,'S':0})

    # Prepare data and split into Train(80%) & Test(20%) 

    Y_feature = customer_data['target']
    customer_data.drop('target', axis=1,inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(
        customer_data, 
        Y_feature, 
        test_size=0.20,
        random_state=5)


    # Apply KNN Imputer on Training data, build model and transform x-train and x-test
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    imputer.fit(x_train)
    x_train = imputer.transform(x_train)
    x_test = imputer.transform(x_test)

    # Train RF Classifier
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(x_train, y_train)

    # Evaluate Model
    predictions = clf.predict(x_test)
    print('F1 score:', f1_score(y_test, predictions))
    print('Recall:', recall_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions))
    print('roc_auc_score:', roc_auc_score(y_test, predictions))

    # Save Model
    joblib.dump(clf, os.path.join(args.model_dir, "RF_Model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    rf_model = joblib.load(os.path.join(model_dir, "RF_Model.joblib"))
    return rf_model
