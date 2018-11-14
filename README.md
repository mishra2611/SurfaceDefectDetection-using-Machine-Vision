# SurfaceDefectDetection-using-Machine-Vision

The project consists of three files which are used for defect identification and localization:
1. batch_processing.py: this contains the model and code for batch processing of data that can handle large datasets
2. MY_Generator.py: this file consists of the file format in which the input and label files are being picked up for processing
3. predict.py: this file contains a preliminary code to load model and predict its outcome for defect localization
4. The files metrics.py contains the function definition of the metrices used for comparision, which is IOU, recall, precision, accuracy and f1score
5. The files resnet.py and segnet.py contains the models for ResNet and SegNet respectively.
6. The files deeplab.py, load_weights.py and extract_weights.py correspond to loading and extracting weights using DeepLab.


Implementation of this Paper :- http://erk.fe.uni-lj.si/2017/papers/racki(towards_surface).pdf is done in the file fccn.py. 
