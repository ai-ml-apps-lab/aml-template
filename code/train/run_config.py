from azureml.core.compute import ComputeTarget
from azureml.train.estimator import Estimator

def main(workspace):

    compute_target = ComputeTarget(workspace=workspace, name="githubcluster")

    script_params = {
        "--kernel": "linear",
        "--penalty": 1.0
    }

    estimator = Estimator(
        source_directory="code/train",
        entry_script="train.py",
        script_params=script_params,
        compute_target=compute_target,
        pip_packages=[
            "scikit-learn",
            "pandas",
            "matplotlib"
        ]
    )

    return estimator
