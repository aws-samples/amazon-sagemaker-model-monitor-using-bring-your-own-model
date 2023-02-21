# SageMaker Model Monitor - Bring your own model

In the repository we present a comprehensive example of how you can leverage the Amazon SageMaker Model Monitor capabilities when you are developing and deploying your models using your own environment (cloud or on-prem) and are looking to SageMaker specifically to help you with monitoring your models.

If you are also training/deploying your models using Amazon SageMaker, then check out the examples in the [aws examples repository](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker_model_monitor) or the blogpost [Automate model retraining with Amazon SageMaker Pipelines when drift is detected](https://aws.amazon.com/blogs/machine-learning/automate-model-retraining-with-amazon-sagemaker-pipelines-when-drift-is-detected/) and its accompanying [GitHub repository](https://github.com/aws-samples/amazon-sagemaker-drift-detection).

In this example we demonstrate how to monitor your model for data quality, model quality, data bias and model explainability changes. 


## Requirements
- Python 3.8+
- pip

## Creating a Python Environment
It is recommended to use a virtual environment to manage the project dependencies. To create a virtual environment, follow these steps:

1. Open your terminal and navigate to the project directory
2. Run the following command to create a virtual environment:
    ```
    python3 -m venv env
    ```
3. Activate the virtual environment:
    - On Windows:
        ```
        env\Scripts\Activate
        ```
    - On Linux/Mac:
        ```
        source env/bin/activate
        ```

## Installing the Dependencies

1. Navigate to the project directory
2. Install the dependencies by running the following command:
    ```
    pip install -r requirements.txt
    ```

## Running the Project

This example is orchestrated by the running of the numbered Jupyter Notebooks that can be found on the root directory of this project. 

#### Setup

In this example we will be using the AWS cloud and therefore we need access to an AWS environment. At this point we assume that you have access to such and account and you have successfully setup a [named profile for the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html). 
Then open the `config.ini.example` file and complete the profile-name parameter with the cli profile name you wish to use. Also, update the IAM role name you will be using to perform the actions. 
Finally, rename the file and remove the `.example` from the name so that it becomes `config.ini`. 
The purpose of this file is to save information like profile name and S3 file locations instead of having to hardcode into our code in multiple locations. Once you reach this step you no longer need to open or manually change this file again.

Before you start you can optionally run the script `prep.py`. This script is preprocessing the data and trains an xgboost model, then saves both data and models to local storage as these will be needed later on. Note that there are two models generated here. 
A preprocessing scikit learn model and the machine learning model, xgboost. These models are provided in the repo to help you get started with the example faster without possible delays of other system dependencies but you are welcome to run this file so you get started from scratch. 

#### Baselining

The first activity required is to run the baselining jobs. For every type of monitoring that we want to perform, we need first to generate the baseline against which we will compare the live data & predictions that we see to determine if any type of drift was detected.

These actions as performed with the use of the SageMaker SDK and are described in the `01_create_baselines.ipynb`. Running that notebook end to end will generate these baselines and will also keep track of the s3 locations of the baselines in the config.ini file. 

The different baselining jobs are run as SageMaker Processing Jobs and this is why you should be able to see these jobs appear in the AWS console under SageMaker > processing jobs. Triggering such a baselining job can be achieved using the SageMaker SDK and the code will look similar in all four types of baselining and similar to the below:

```python
my_default_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)

my_default_monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri+'/train_data_no_target.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_results_uri,
    wait=True,
)
```

You may also notice that at this section of the code we are creating a SageMaker Model. This is used from the baselining job (as well as the monitoring that is run in a next section) for the explainability and bias drift detections. It is necessary as the SageMaker service Clarify that we are making use of, needs to be able to make use of the actual model to generate predictions during drift detection phase. Internally, a SageMaker endpoint will be created to serve these requests without affecting the production/live version of the model. As soon as the baselining process is finished, these endpoints are automatically removed so there is no risk of running into unexpected recurring costs. 

#### Monitoring checks

Start with opening an interactive python session and running 
```python 
import generate_altered
generate_altered.main()
```

This will generate some synthetic data that simulate data that you might observe when running your application.

The four next notebooks are responsible to perform the respective monitoring checks as their respective name suggests. 

For more details on the different jobs, please open the respective notebook file or read the description of [Monitoring Checks](MonitoringChecks.md).


#### Creating artificial drift

The `05_orchestrate_drift.ipynb` notebook will run a number of different alterations of data and then respective checks. This is done to simulate different scenarios that might occur for different reasons to your data. 

Once you are familiar with this repository, you are welcome to define your own scenario by adding or alterning an existing scenario in `generate_altered.py` file.

Please note that this notebook is also making use of sagemaker experiments which is used solely to help us organise the different experiments and allow us to more easily manage those, as it will become apparent in the next section. 

#### Evaluating results of experiments

Evaluating the results of these experiments is done in `06_evaluate_results.ipynb` where the monitoring results are picked up and displayed in plots. Note here, that for each scenario, you are asked to provide a "TrianComponentName" arn which is used to query & retrieve the constraint_violations s3 uris. If you prefer to not use the sagemaker experiments capabilities, you can also directly pass the s3 uris with the contraints violations. 

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

----

Author: Georgios Schinas