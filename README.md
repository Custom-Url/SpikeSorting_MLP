# SpikeSorting_MLP

Spike Sorting to identify neuron firings in large datasets containing a range of noise values.

The code uses TensorFlow to implement a set of MLP models. Due to the range of SNR values across the data sets, a model is trained for each data set. This allows for unique filter parameters depending on the noise level. Filtering is implemented using a Butterworth filter followed by a Savitzky-Golay for data smoothing. As each data set has individual filtering parameters, a set of filtered data is produced for D1 for each filter. This ensures that peak shapes in the training data matches the peaks shapes in the classification data as closely as possible. Peaks are identified and indexed in each of the filtered data sets and, again due to the range of SNR values, detection thresholds are varied for each data set. D1 indexing is updated so that the index is moved to peak which matches the indexing location of the classification data. The data points around each peak are stacked together to produce an array containing the spike shape. This array is used for training and classification. D1 is split 80:20 to produce training and verification data for the MLPs. The MLPs are then initialised and trained on the D1 data. Finally, MLPs are used to identify peak classes and respective Class and Index arrays are exported to .mat files.

The code produces high precision and accuracy for D2, D3, and D4. Recall remains high, around 80% for D5 and D6 but precision decreases as the SNR approaches 0dB. Visual validation of data filtering was possible for datasets with high SNR however as peaks became more indistinguishable from noise this was more challenging. This is likely a factor in the reduced precision shown in the results for D5 and D6. Further development and testing of filter parameters could show improvement for low SNRs. Overall, the filtering and peak detection, and the MLP peak classification performed well.
