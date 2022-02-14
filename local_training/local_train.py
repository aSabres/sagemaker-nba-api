import argparse
import os

import pandas
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# When the training job finishes, the container will be deleted including its file system with exception of the
# /opt/ml/model and /opt/ml/output folders.
# Use /opt/ml/model to save the model checkpoints.
# These checkpoints will be uploaded to the default S3 bucket.

# SageMaker default SM_MODEL_DIR=/opt/ml/model
if os.getenv("SM_MODEL_DIR") is None:
    os.environ["SM_MODEL_DIR"] = os.getcwd() + '/model'

# SageMaker default SM_OUTPUT_DATA_DIR=/opt/ml/output
if os.getenv("SM_OUTPUT_DATA_DIR") is None:
    os.environ["SM_OUTPUT_DATA_DIR"] = os.getcwd() + '/output'

# SageMaker default SM_CHANNEL_TRAINING=/opt/ml/input/data/training
if os.getenv("SM_CHANNEL_TRAINING") is None:
    os.environ["SM_CHANNEL_TRAINING"] = os.getcwd() + '/data'


def dnn_training(args):

    print("Start Training:")

    print("Training dataset: ", args.train + "/" + args.dataset_file)

    dataset = pandas.read_csv(args.train + "/" + args.dataset_file)

    print(dataset['IS_ALL_STAR'].value_counts())

    # Fit the classifier model
    x_train = dataset.loc[:, ~dataset.columns.isin(['W_PCT', 'MIN', 'IS_ALL_STAR'])]
    y_train = dataset.loc[:, 'IS_ALL_STAR']

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=16))
    #model.add(Dense(22, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))




    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # early stopping callback
    # This callback will stop the training when there is no improvement in
    # the validation loss for args.es_patience consecutive epochs.
    es = EarlyStopping(monitor='val_accuracy',
                                       mode='max', # don't minimize the accuracy!
                                       patience=args.es_patience,
                                       restore_best_weights=True)

    model.summary()

    # train the model
    hist = model.fit(x_train, y_train, validation_data=(np.asarray(x_valid).astype(np.float32), np.asarray(y_valid).astype(np.float32)),
                     shuffle=True, callbacks=[es], epochs=args.epochs, batch_size=args.batch_size)


    # make a prediction
    x_test = np.array([[105.9,105.2,0.7,0.169,2.24,19.2,0.073,0.21,0.142,8.6,8.6,0.558,0.587,0.197,74.98,0.15]])
    predictions = model.predict(np.asarray(x_test).astype(np.float32))
    print(predictions)

    # outputs are in type int, convert them to bool
    #y_test = (y_test > 0.5)
    predictions = (predictions > 0.5)
    print(predictions)

    return model
    # make a prediction
    # predictions = model.predict(x_test)

    # outputs are in type int, convert them to bool
    # y_test = (y_test > 0.5)
    # predictions = (predictions > 0.5)

    # Prepare y_prediction and y_truth for future inspaction
    # y_prediction = predictions # Predict test data
    # y_truth = y_test.to_numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SM default params
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--dataset_file', type=str, default='train_set_adv_pd.csv')

    # hyperparameters are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=2) # 5
    parser.add_argument('--batch_size', type=int, default=250) # 250
    parser.add_argument('--es_patience', type=int, default=40)
    # parser.add_argument('--learning_rate', type=float, default=0.01)
    # parser.add_argument('--drop_rate', type=float, default=0.2)
    # parser.add_argument('--dense_hidden', type=int, default=128)

    args, _ = parser.parse_known_args()
    print(args)

    # training the dnn
    model = dnn_training(args)

    # save the model
    # it seems that it's important to have a numerical name for your folder:
    model_path = args.model_dir + '/1'
    #model_path = r'C:\Users\Asaf\PycharmProjects\sagemaker-nba-api\local_training\model\1'
    print('The model will be saved at :', model_path)
    model.save(model_path)
    print('model saved')