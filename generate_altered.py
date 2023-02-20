import sagemaker
import boto3
from sagemaker.s3 import S3Downloader
import pandas as pd
from io import StringIO
import random
import enum
from utils import get_baseline_uri, save_dataset, get_aws_profile_name, get_aws_iam_role
import prep

# ----- Initial variables -----
# def init_variables():
    # global sess, sm, iam, role, bucket, prefix, region
LOCAL_EXECUTION = True

if LOCAL_EXECUTION:
    sess = boto3.Session(profile_name=get_aws_profile_name())
    sm = sess.client("sagemaker")
    iam = sess.client('iam')
    role = iam.get_role(RoleName=get_aws_iam_role())['Role']['Arn']
else:
    sess = boto3.Session()
    sm = sess.client("sagemaker")
    role = sagemaker.get_execution_role()

sagemaker_session = sagemaker.Session(boto_session=sess)
bucket = sagemaker_session.default_bucket()
prefix = "model-monitor-bring-your-own-model/"
region = sess.region_name
# ------------


def load_data(dataset_name='model-explainability-baseline-data', sagemaker_session=sagemaker_session)-> pd.DataFrame:
    data_string = S3Downloader().read_file(get_baseline_uri(dataset_name), sagemaker_session=sagemaker_session)
    data_string = StringIO(data_string)
    # df = pd.read_csv(data_string, index_col="Unnamed: 0")
    df = pd.read_csv(data_string)
    return df


def save_data(df:pd.DataFrame, dataset_name="data-quality-modified-data", data_prefix="data-quality/modified_input_data", sagemaker_session=sagemaker_session)->None:
    filename = f"data/{dataset_name.replace('-','_')}.csv"
    df.to_csv(filename, index=False, header=True)
    data_prefix = prefix + data_prefix
    modified_data_uri = sagemaker_session.upload_data(
        path=filename, bucket=bucket, key_prefix=data_prefix
    )
    save_dataset(dataset_name, modified_data_uri)


def default_transform(df:pd.DataFrame)->pd.DataFrame:

    # ammount increase by random amount between 0 and 10k
    df["amount"] = df.amount.apply(lambda x: x + random.randint(0,10000))

    # a random 30% of people have less savings bringing them to one category below where they were
    df["savings"] = df.savings.apply(lambda x: x - 1 * (x>1) * (random.random()>0.7))

    df = prep.predict(df)

    # a random 30% of predictions will be wrong
    df["credit_risk"] = df["prediction"].apply(lambda x: (1-x) if random.random()>0.7 else x)
    return df


def concept_drift(df:pd.DataFrame)->pd.DataFrame:
    # Scenario: In 2023 amount level of 1000 or less is always considered NOT credit worthy in ground truth
    df = prep.predict(df)

    # a random 30% of predictions will be wrong
    df["credit_risk"] = df["prediction"].apply(lambda x: (1-x) if random.random()>0.7 else x)

    #
    df["credit_risk"].loc[df["amount"] <= 1000] = 0
    
    return df


def label_drift(df:pd.DataFrame)->pd.DataFrame:
    # Scenario: In 2023 we receive more applications that have been predicted as credit worthy
    df = prep.predict(df)
    df["credit_risk"] = df["credit_risk"].apply(lambda x: 5*[x] if (x == 1) else x)
    df = df.explode('credit_risk', ignore_index=True)

    # # a random 10% of predictions will be wrong
    # df["credit_risk"] = df["prediction"].apply(lambda x: (1-x) if random.random()>0.9 else x)
    return df


def feature_drift(df:pd.DataFrame)->pd.DataFrame:
    # Scenario: In 2023 the savings of applicants significantly drops

    # a random 80% of people have less savings bringing them to one category below where they were
    df["duration"] = df["duration"] + int(2*df["duration"].std())
    df = prep.predict(df)
    # a random 30% of predictions will be wrong
    df["credit_risk"] = df["prediction"].apply(lambda x: (1-x) if random.random()>0.7 else x)
    return df


def feature_drift_systematic(df:pd.DataFrame)->pd.DataFrame:
    # Scenario: In 2023 the savings of applicants significantly drops

    # a random 80% of people have less savings bringing them to one category below where they were
    df["duration"] = 1
    df = prep.predict(df)
    # a random 10% of predictions will be wrong
    # df["credit_risk"] = df["prediction"].apply(lambda x: (1-x) if random.random()>0.9 else x)
    return df


def explainability_drift(df:pd.DataFrame)->pd.DataFrame:
    # Scenario: In 2023 due to change in X feature, the feature becomes overall more important

    df["duration"] = df["duration"] + int(2*df["duration"].std())

    # a random 80% of people have less savings bringing them to one category below where they were
    df["savings"] = df.savings.apply(lambda x: x - 1 * (x>1) * (random.random()>0.2))

    df["status"] = df.status.apply(lambda x: 4 if x<3 else x) #changing status to primarily 4 which means we have long salary information

    df = prep.predict(df)
    # a random 30% of predictions will be wrong
    df["credit_risk"] = df["prediction"].apply(lambda x: (1-x) if random.random()>0.7 else x)
    return df


def bias_drift(df:pd.DataFrame) -> pd.DataFrame:
    # Scenario: In 2023 the model predicted that if foreign_worker then it is not credit worthy
    df = prep.predict(df)
    # df["prediction"].loc[df["foreign_worker"] ==1] = 0

    # removing all but one positive foreign worker predictions
    filter_fw = ((df['foreign_worker'] ==1) & (df['prediction'] ==1))
    for i,v in df.iterrows():
        if (v["foreign_worker"] == True) and (v["prediction"]==1):
            filter_fw[i]= False
            break
    df = df[~filter_fw]

    df["foreign_worker"] = df["foreign_worker"].apply(lambda x: 5*[x] if (x == 1) else x)
    df = df.explode('foreign_worker', ignore_index=True)

    df["foreign_worker"] = df["foreign_worker"].apply(lambda x: 100*[x])
    df = df.explode('foreign_worker', ignore_index=True)

    # adding more positive predictions for non-foreign to influence the ratio more
    filter_nfw = ((df['foreign_worker'] ==2) & (df['prediction'] ==1))
    df.loc[filter_nfw, "foreign_worker"] = df["foreign_worker"].apply(lambda x: 10*[x])
    df = df.explode('foreign_worker', ignore_index=True)

    # a random 10% of predictions will be wrong
    # df["credit_risk"] = df["prediction"].apply(lambda x: (1-x) if random.random()>0.9 else x)
    return df


class TransformsEnum(enum.Enum):
    DEFAULT = default_transform
    CONCEPT_DRIFT = concept_drift
    LABEL_DRIFT = label_drift
    FEATURE_DRIFT = feature_drift
    FEATURE_DRIFT_SYSTEMATIC = feature_drift_systematic
    EXPLAINABILITY_DRIFT = explainability_drift
    BIAS_DRIFT = bias_drift
    

def main(transform: TransformsEnum = TransformsEnum.DEFAULT):
    df = load_data()

    df = transform(df)

    dq_data = df.drop(['prediction_probability', 'prediction', "credit_risk"], axis=1)
    mq_data = df
    me_data = df.drop(['prediction_probability', 'credit_risk'], axis=1).rename({"prediction":"credit_risk"}, axis=1) # during bias and explainability we are interested in what the model has predicted, not the ground truth
    mb_data = me_data
    
    save_data(dq_data, "data-quality-modified-data", "data-quality/modified_input_data")
    save_data(mq_data, "model-quality-modified-data", "model-quality/input_data")
    save_data(me_data, "model-explainability-modified-data",  "model-explainability/input_data")
    save_data(mb_data, "model-bias-modified-data",  "model-bias/input_data")


if __name__ == "__main__":
    main()