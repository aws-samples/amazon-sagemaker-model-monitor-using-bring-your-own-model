
# Download data from  "s3://sagemaker-sample-files/datasets/tabular/uci_statlog_german_credit_data/SouthGermanCredit.asc"
from json.tool import main
import pandas as pd
import pandas as pd
import os
import numpy as np
import tarfile
import sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
import xgboost
import pickle as pkl


DATA_LOCATION_PREFIX = "data/"
PROCESSED_DATA_PREFIX = "processed/"
MODEL_LOCATION_PREFIX = "trained/"
RAW_DATA_LOCATION = "raw/SouthGermanCredit.asc"
TEST_DATA = "test.csv"
TRAIN_DATA = "train.csv"
TRAIN_FEATURES = "train_features.csv"
TRAIN_LABELS= "train_labels.csv"
VAL_FEATURES= "val_features.csv"
VAL_LABELS = "val_labels.csv"
PROCESSOR_MODEL_NAME = "model.joblib"
PROCESSOR_MODEL_TAR_NAME = "model.tar.gz"

os.makedirs(DATA_LOCATION_PREFIX, exist_ok = True)
os.makedirs(DATA_LOCATION_PREFIX+PROCESSED_DATA_PREFIX, exist_ok = True)
os.makedirs(DATA_LOCATION_PREFIX+MODEL_LOCATION_PREFIX, exist_ok = True)

credit_columns = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

def unpack_data():
    training_data = pd.read_csv(
        DATA_LOCATION_PREFIX+RAW_DATA_LOCATION,
        names=credit_columns,
        header=0,
        sep=r" ",
        engine="python",
        na_values="?",
    ).dropna()

    test_data = training_data.sample(frac=0.1)
    test_data = test_data.drop(["credit_risk"], axis=1)
    test_filename = DATA_LOCATION_PREFIX + TEST_DATA
    test_columns = [
        "status",
        "duration",
        "credit_history",
        "purpose",
        "amount",
        "savings",
        "employment_duration",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "present_residence",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "number_credits",
        "job",
        "people_liable",
        "telephone",
        "foreign_worker",
    ]
    test_data.to_csv(test_filename, index=False, header=True, columns=test_columns, sep=",")

    # prepare raw training data
    train_filename = DATA_LOCATION_PREFIX + TRAIN_DATA
    training_data.to_csv(train_filename, index=False, header=True, columns=credit_columns, sep=",")

def preprocess(train_test_split_ratio=.2):

    # Read input data into a Pandas dataframe.
    input_data_path = DATA_LOCATION_PREFIX + TRAIN_DATA
    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path, names=None, header=0, sep=",")

    # Defining one-hot encoders.
    print("performing one hot encoding")
    transformer = make_column_transformer(
        (
            OneHotEncoder(sparse=False),
            [
                "credit_history",
                "purpose",
                "personal_status_sex",
                "other_debtors",
                "property",
                "other_installment_plans",
                "housing",
                "job",
                "telephone",
                "foreign_worker",
            ],
            
        ),
        remainder="passthrough",
    )

    print("preparing the features and labels")
    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]

    print("building sklearn transformer")
    featurizer_model = transformer.fit(X)
    features = featurizer_model.transform(X)
    labels = LabelEncoder().fit_transform(y)

    # Splitting.
    split_ratio = train_test_split_ratio
    print("Splitting data into train and validation sets with ratio {}".format(split_ratio))
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=split_ratio, random_state=0
    )

    print("Train features shape after preprocessing: {}".format(X_train.shape))
    print("Validation features shape after preprocessing: {}".format(X_val.shape))

    # Saving outputs.
    train_features_output_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + TRAIN_FEATURES
    train_labels_output_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + TRAIN_LABELS

    val_features_output_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + VAL_FEATURES
    val_labels_output_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + VAL_LABELS

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    pd.DataFrame(y_train).to_csv(train_labels_output_path, header=False, index=False)

    print("Saving validation features to {}".format(val_features_output_path))
    pd.DataFrame(X_val).to_csv(val_features_output_path, header=False, index=False)

    print("Saving validation labels to {}".format(val_labels_output_path))
    pd.DataFrame(y_val).to_csv(val_labels_output_path, header=False, index=False)

    # Saving model.
    model_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + PROCESSOR_MODEL_NAME
    model_output_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + PROCESSOR_MODEL_TAR_NAME

    print("Saving featurizer model to {}".format(model_output_path))
    joblib.dump(featurizer_model, model_path)
    tar = tarfile.open(model_output_path, "w:gz")
    tar.add(model_path, arcname=PROCESSOR_MODEL_NAME)
    tar.close()


def train():

    train_features_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + TRAIN_FEATURES
    train_labels_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + TRAIN_LABELS

    val_features_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + VAL_FEATURES
    val_labels_path = DATA_LOCATION_PREFIX + PROCESSED_DATA_PREFIX + VAL_LABELS

    print("Loading training dataframes...")
    df_train_features = pd.read_csv(train_features_path)
    df_train_labels = pd.read_csv(train_labels_path)

    print("Loading validation dataframes...")
    df_val_features = pd.read_csv(val_features_path)
    df_val_labels = pd.read_csv(val_labels_path)

    X = df_train_features.values
    y = df_train_labels.values

    val_X = df_val_features.values
    val_y = df_val_labels.values

    dtrain = xgboost.DMatrix(X, label=y)
    dval = xgboost.DMatrix(val_X, label=val_y)

    watchlist = [(dtrain, "train"), (dval, "validation")]

    params = {
        "max_depth": "5",
        "eta": "0.1",
        "gamma": "4",
        "min_child_weight": "6",
        "silent": "1",
        "objective": "binary:logistic",
        "subsample": "0.8",
        "eval_metric": "auc",
        "early_stopping_rounds": "20",
    }

    bst = xgboost.train(
        params=params, dtrain=dtrain, evals=watchlist, num_boost_round=100
    )

    model_dir = DATA_LOCATION_PREFIX + MODEL_LOCATION_PREFIX
    pkl.dump(bst, open(model_dir + "/model.bin", "wb"))

    yhat = bst.predict(xgboost.DMatrix(val_X))

    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precision, recall, thresholds = precision_recall_curve(val_y, yhat)
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.nanargmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

def inference(filename, output_filename):

    input_data = pd.read_csv(filename)

    input_data = predict(input_data=input_data)

    if output_filename:
        input_data.to_csv(output_filename)
    else:
        print("Not saving data")
    print("finished")


def predict(input_data):
    threshold = 0.56 # computed during training
    preprocessor =  joblib.load(DATA_LOCATION_PREFIX+PROCESSED_DATA_PREFIX+PROCESSOR_MODEL_NAME)
    model = pkl.load(open(DATA_LOCATION_PREFIX+MODEL_LOCATION_PREFIX+"/model.bin", "rb")) 
    
    features = preprocessor.transform(input_data)
    array = np.array(features)
    array = xgboost.DMatrix(array)

    predictions = model.predict(array)
    input_data['prediction_probability'] = list(predictions)
    input_data['prediction'] = input_data['prediction_probability'].apply(lambda x: 1 if x >= threshold else 0)

    return input_data


if __name__ == "__main__":

    unpack_data()
    preprocess()

    train()

    inference(filename=DATA_LOCATION_PREFIX + TEST_DATA, output_filename=None)
    inference(filename=DATA_LOCATION_PREFIX + TRAIN_DATA, output_filename=DATA_LOCATION_PREFIX+"train_data_with_prediction.csv")

    train_df = pd.read_csv(DATA_LOCATION_PREFIX + TRAIN_DATA)
    train_df.drop("credit_risk", axis=1, inplace=True)
    train_df.to_csv(DATA_LOCATION_PREFIX+"train_data_no_target.csv", index=False)