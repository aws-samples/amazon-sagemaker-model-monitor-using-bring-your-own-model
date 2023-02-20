# Monitoring Checks

All four types of checks for drift will be run using the SageMaker Model Monitor managed containers (so no custom code is required for the monitoring). Since we are not utilising the rest of the SageMaker suite of services, we are not able to schedule the monitoring jobs to run automatically so we will need to trigger them on-demand and use SageMaker Processing Jobs to do so. 
Of course, when implementing this for your scenario, you will need to schedule these jobs using the tool of your choice. 

In all below scenarios, the detected violations are written in the constraint_violations.json file, output of the Processing Jobs. 

## Data Quality Drift

Data Quality drift is implemented in the 02_data_quality_checks.ipynb notebook. Looking at this notebook, you will realise that the process of running the data quality check is fairly simple and there are only a couple points to note. 
The code that will perform the drift detection lives inside the SageMaker managed container for model monitoring and you can retrieve it with
```python
model_monitor_container_uri = sagemaker.image_uris.retrieve(
            framework="model-monitor",
            region=region,
            version="latest",
        )
```
Then, since we are not using the built in triggering mechanism from an endpoint or batch transform, we need to instruct the container what exactly to do and we achieve this with the use of environment variables. This is common methodology across the four types of drift. For data quality this takes the form of: 

```python
env = {
    "dataset_format": json.dumps(dataset_format),
    "dataset_source": "/opt/ml/processing/input/baseline_dataset_input",
    "output_path": "/opt/ml/processing/output",
    "publish_cloudwatch_metrics": "Disabled",
    "baseline_constraints": "/opt/ml/processing/baseline/constraints/constraints.json",
    "baseline_statistics": "/opt/ml/processing/baseline/stats/statistics.json",
}
```
Given the above two snippets, we can start a processing job as below, which performs a data quality check for us. Also note, that we need to pass in as inputs the modified or “live“ dataset as well as the baseline files. 
```python
monitor_analyzer = Processor(
    image_uri=model_monitor_container_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    base_job_name=f"model-monitor-byom",
    sagemaker_session=sagemaker_session,
    max_runtime_in_seconds=1800,
    env=env,
)

monitor_analyzer.run(
    inputs=[ProcessingInput(
                source=get_dataset_uri('data-quality-modified-data'),
                destination="/opt/ml/processing/input/baseline_dataset_input",
                input_name="baseline_dataset_input",),
            ProcessingInput(
                source=get_baseline_uri('data-quality-constraints'),
                destination="/opt/ml/processing/baseline/constraints",
                input_name="constraints",
                ),
            ProcessingInput(
                source=get_baseline_uri('data-quality-statistics'),
                destination="/opt/ml/processing/baseline/stats",
                input_name="baseline",
                ),
            ],
    outputs=[
        ProcessingOutput(
                    source="/opt/ml/processing/output",
                    output_name="monitoring_output",
                )
    ],
)
```

## Model Quality check

Model Quality drift is implemented in the 03_model_quality_checks.ipynb notebook and is almost identical to the above process, with the difference of the needed environment variables. Note that the environment file for model quality checks is a bit more elaborate.

```python
env = {
    "dataset_format": json.dumps(dataset_format),
    "dataset_source": "/opt/ml/processing/input/baseline_dataset_input",
    "output_path": "/opt/ml/processing/output",
    "publish_cloudwatch_metrics": "Disabled",
    "analysis_type":"MODEL_QUALITY",
    "problem_type":'BinaryClassification',
    "inference_attribute": "prediction", # The column in the dataset that contains predictions.
    "probability_attribute": "prediction_probability", # The column in the dataset that contains probabilities.
    "ground_truth_attribute": "credit_risk",
    "baseline_constraints": "/opt/ml/processing/baseline/constraints/constraints.json",
    "baseline_statistics": "/opt/ml/processing/baseline/stats/statistics.json",
}
```

## Explainability & Bias checks

In the 04_model_explainability_checks.ipynb and 04_model_bias_checks.ipynb notebook we are performing the model explainability drift and bias drift checks respectively. For this one, we use a different managed container. We use the one provided by [SageMaker Clarify](https://aws.amazon.com/sagemaker/clarify/?sagemaker-data-wrangler-whats-new.sort-by=item.additionalFields.postDateTime&sagemaker-data-wrangler-whats-new.sort-order=desc). 

```python
model_explainability_container_uri = sagemaker.image_uris.retrieve(
            framework="clarify",
            region=region,
            version="latest",
        )
```

Following the same pattern as before, we need to again set the environment variables, but in this case with the following

```python
env = {
    "baseline_constraints": "/opt/ml/processing/baseline/constraints/analysis.json",
}
```

In both cases, the container will know if it needs to perform explainability or bias checks based on. the content of the baseline file. 
