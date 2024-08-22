import tensorflow as tf
import scipy.signal as signal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat, savemat

'''
BEFORE RUNNING CODE:
Open Terminal and run the following command to install TensorFlow

pip install tensorflow
'''

# Configuration
config = {
    'data_files': ['D1.mat', 'D2.mat', 'D3.mat', 'D4.mat', 'D5.mat', 'D6.mat'],
    'filter_params': {
        'D2': (20, 25000, 60, 2, 2),
        'D3': (20, 25000, 60, 2, 2),
        'D4': (20, 25000, 60, 2, 2),
        'D5': (10, 25000, 60, 2, 2),
        'D6': (10, 25000, 30, 2, 2),
    },
    'threshold_params': {
        'D2': (0.4, 60),
        'D3': (0.4, 60),
        'D4': (0.75, 60),
        'D5': (1.0, 60),
        'D6': (2.0, 60),
    },  
    'frame_width': 60,
    'test_size': 0.2,
    'epochs': 100,
    'batch_size': 64,
}

# # Function Definitions:
# Build MLP Model
def create_and_train_model(x_train, y_train, x_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(60,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_data=(x_test, y_test))
    return model

def stackFrames(data, index, width):
# Creates 'frame' around peak index for MLP training and classification
    stackedFrame = []
    # Padding for half of the width
    padding = np.zeros(width // 2)
    # Pad the data and stack frames
    padded_data = np.concatenate((padding, data, padding))
    for i in index:
        stackedFrame.append(padded_data[i:i+width])
    return np.vstack(stackedFrame)

def findPeakCenter(data, Index,frameWidth):
# Finds the centre of each labelled peak so training index matches classification
    updatedIndex = []
    for peakIndex in Index:
        frame = data[peakIndex - frameWidth//2 : peakIndex + frameWidth//2]
        localMaxIndex = np.argmax(frame)
        centreIndex = peakIndex - frameWidth//2 + localMaxIndex
        updatedIndex.append(centreIndex)
    return updatedIndex

def butterworthFilter(data, cutoffFrequency, samplingRate, windowLength, savgolOrder, butterworthOrder):
    # Normalise cutoff frequency
    cutoffNorm = cutoffFrequency / (samplingRate / 2)
    # Apply Butterworth filter
    b, a = signal.butter(butterworthOrder, cutoffNorm, btype='high', analog=False)
    butterworth_filtered = signal.filtfilt(b, a, data)
    # Apply SavGol smoothing filter
    savgolFiltered = signal.savgol_filter(butterworth_filtered, windowLength, savgolOrder)
    return savgolFiltered

def trainingData(d1Filtered, Class):
    # Split processed D1 into training and verification/test data
    xTrainSplit, xTestSplit, yTrainSplit, yTestSplit = train_test_split(d1Filtered, Class, test_size=0.2)
    # Initialise scaler
    scaler = StandardScaler()
    # Apply scaler to training and verification/test data
    xTrainSplit = scaler.fit_transform(xTrainSplit)
    xTestSplit = scaler.transform(xTestSplit)  
    return xTrainSplit, xTestSplit, yTrainSplit, yTestSplit, scaler

# Load the MATLAB file for d1 for MLP training
mat_d1 = loadmat('D1.mat', squeeze_me=True)
d1 = mat_d1['d']
Index1 = mat_d1['Index']
Class = mat_d1['Class'] - 1  # -1 to update class to zero indexing for processing

# Load the MATLAB files for d2, d3, d4, d5, d6
datasets = {}
for file in config['data_files'][1:]:
    mat = loadmat(file, squeeze_me=True)
    datasets[file[:-4]] = {'d': mat['d']}

# Iterate over datasets
for dataset, data in datasets.items():
    # Filtering
    filteredData = butterworthFilter(data['d'], *config['filter_params'][dataset])
    d1Filtered = butterworthFilter(d1, *config['filter_params'][dataset])
    
    # Peak indexing
    peaks, _ = signal.find_peaks(filteredData, height=config['threshold_params'][dataset][0], distance = config['threshold_params'][dataset][1])
    Index1 = findPeakCenter(d1Filtered, Index1, config['frame_width'])

    # Frame stacking
    stackedFrames = stackFrames(filteredData, peaks, config['frame_width'])
    d1Stacked = stackFrames(d1Filtered, Index1, config['frame_width'])

    xTrainSplit, xTestSplit, yTrainSplit, yTestSplit, scaler = trainingData(d1Stacked, Class)

    # Build the MLP model 
    model = create_and_train_model(xTrainSplit, yTrainSplit, xTestSplit, yTestSplit)

    # Use MLP to classify the dataset
    predicted_class = np.argmax(model.predict(scaler.transform(stackedFrames)), axis=1) + 1

    # Save results
    savemat(f"Result_{dataset}.mat", {"Index": peaks, "Class": predicted_class})
