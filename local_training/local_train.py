from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf

#from sklearn.model_selection import train_test_split

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
    
    print("Start Training")
    
    # Fit the classifier model
    # x_train = train_set_pd.iloc[:,0: -1]
    # y_train = train_set_pd.iloc[:,-1]

    #x_train, x_valid, y_train, y_valid = train_test_split(train_set_pd.iloc[:,0: -1], train_set_pd.iloc[:,-1], test_size=0.33, random_state=42)


    # hide the real values from the model
    x_test = test_set.iloc[:,5: -2]
    y_test = test_set.iloc[:,-1]    
    
    print("Dataset: ", os.listdir(path=args.train))
    
    model = Sequential() 
    model.add(Dense(36, activation='relu', input_dim=36))
    model.add(Dense(22, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) 

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # early stopping callback
    # This callback will stop the training when there is no improvement in  
    # the validation loss for 100 consecutive epochs.  
    es = EarlyStopping(monitor='val_accuracy', 
                                       mode='max', # don't minimize the accuracy!
                                       patience=40,
                                       restore_best_weights=True)

    model.summary()

    # train the model
    hist = model.fit(x_train, y_train, validation_data=(np.asarray(x_valid).astype(np.float32), np.asarray(y_valid).astype(np.float32)),
                     shuffle=True, callbacks=[es], epochs=250, batch_size=5)


# make a prediction
# predictions = model.predict(x_test)

# outputs are in type int, convert them to bool 
# y_test = (y_test > 0.5)
# predictions = (predictions > 0.5)

# Prepare y_prediction and y_truth for future inspaction
# y_prediction = predictions # Predict test data
# y_truth = y_test.to_numpy()