from azureml.core import ComputeTarget
from azureml.train.estimator import Estimator

def main(workspace):

    compute_target = ComputeTarget(workspace=workspace, name="githubcluster")

    estimator = Estimator(
        source_directory="code/train",
        entry_script="train.py",
        compute_target=compute_target,
        script_params={
            "--kernel": "linear",
            "--penalty": 1.0
        },
        pip_packages=[
            "azureml-dataprep[pandas,fuse]",
            "scikit-learn",
            "pandas",
            "matplotlib"
        ]
    )

    return estimator
