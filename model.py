import json
import numpy as np
import tensorflow.keras as nn
from sklearn.model_selection import train_test_split

#function to read back in json data and split into train, validation, and test data
def prepare_dataset(dataPath):
    with open(dataPath + 'data.json') as f:
        data = json.load(f)
    
    input = np.array(data['mfcc'])
    target = np.array(data['label'])
    
    inputTrain, inputTest, targetTrain, targetTest = train_test_split(input, 
                                                                        target,
                                                                        test_size = 0.1)
    
    inputTrain, inputValid, targetTrain, targetValid = train_test_split(inputTrain, 
                                                                        targetTrain,
                                                                        test_size = 0.2)
    
    #since mfccs do not have an rgb channel, an extra axis of 1s is added as a placeholder
    inputTrain = inputTrain[..., np.newaxis]
    inputTest = inputTest[..., np.newaxis]
    inputValid = inputValid[..., np.newaxis]
    
    return inputTrain, inputTest, inputValid, targetTrain, targetTest, targetValid

#function to initalize model layers
def build_model(inputShape):
    
    model = nn.Sequential([
        
    nn.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=inputShape),
    nn.layers.MaxPool2D((3,3), strides=(2,2)),
    nn.layers.BatchNormalization(),
    
    nn.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    nn.layers.MaxPool2D((3,3), strides=(2,2)),
    nn.layers.BatchNormalization(),
    
    nn.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    nn.layers.Conv2D(64, (2,2), activation='relu', padding='same'),
    nn.layers.MaxPool2D((2,2), strides=(2,2)),
    nn.layers.BatchNormalization(),

    nn.layers.Flatten(),
    nn.layers.Dense(64, activation='relu'),
    nn.layers.Dropout(0.3),
    
    nn.layers.Dense(10, activation='softmax')
    ])

    return model


if __name__ == "__main__":
    
    inputTrain, inputTest, inputValid, targetTrain, targetTest, targetValid = prepare_dataset(dataPath='preprocessed_data/')
    
    #since inputTrain has 4 dimensions, the first being number of samples, we just want dimensions 2 to 4
    inputShape = (inputTrain.shape[1], inputTrain.shape[2], inputTrain.shape[3])
    model = build_model(inputShape=inputShape)
    
    optimizer = nn.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(inputTrain, targetTrain, validation_data=(inputValid, targetValid), batch_size=32, epochs=50)
    
    #evaluate model on the test set and print out the final accuracy score
    testError, testAccuracy = model.evaluate(inputTest, targetTest, verbose=1)
    print("Accuracy of test set is: " + str(testAccuracy))
    