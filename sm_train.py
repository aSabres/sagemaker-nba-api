import os
from time import gmtime, strftime
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

def sagemaker_estimator(sagemaker_role,code_entry,code_dir, instance_type, instance_count, hyperparameters, metric_definitions):
    sm_estimator = TensorFlow(entry_point=code_entry,
                              source_dir=code_dir,
                              role=sagemaker_role,
                              instance_type=instance_type,
                              instance_count=instance_count,
                              model_dir='/opt/ml/model',
                              hyperparameters=hyperparameters,
                              metric_definitions=metric_definitions,
                              framework_version='2.2',
                              py_version='py38',
                              script_mode=True)
    return sm_estimator


def sagemaker_training(sm_estimator,train_s3,training_job_name):
    sm_estimator.fit(train_s3, job_name=training_job_name, wait=False)
    
if __name__ == '__main__':
    
    session = sagemaker.Session()
    sagemaker_role = get_execution_role()
    
    train_s3 = "s3://asaf-sagemaker-datasets/final_dataset/output_1642495104/part-00000-db74d4ca-2111-4b23-a734-0f2b4ecd417f-c000.csv"
    print(train_s3)
    print(os.getcwd())
    
    code_entry = 'local_train.py'
    code_dir = os.getcwd() + '/local_training/'
    print(code_dir)
    instance_type = 'ml.c5.xlarge'
    instance_count = 1
    hyperparameters = {'epochs': 10,
                       'batch_size': 250,
                       'es_patience': 40}

    metric_definitions = [
        {'Name': 'train:error', 'Regex': 'loss: ([0-9\\.]+)'},
        {'Name': 'validation:error', 'Regex': 'val_loss: ([0-9\\.]+)'},
        {'Name': 'validation:accuracy', 'Regex': 'val_accuracy: ([0-9\\.]+)'}
    ]
    
    sm_estimator = sagemaker_estimator(sagemaker_role, code_entry, code_dir, instance_type, instance_count, hyperparameters, metric_definitions)
        
    # sagemaker training job
    training_job_name = "tf-nba-training-{}".format(strftime("%d-%H-%M-%S", gmtime()))
    sagemaker_training(sm_estimator, train_s3, training_job_name)    