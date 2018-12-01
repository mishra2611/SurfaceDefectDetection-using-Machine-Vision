# SurfaceDefectDetection-using-Machine-Vision

The project consists of three files which are used for defect identification and localization:  
1. batch_processing.py: this contains the model and code for batch processing of data that can handle large datasets.  
2. MY_Generator.py: this file consists of the file format in which the input and label files are being picked up for processing.  
3. predict.py: this file contains a preliminary code to load model and predict its outcome for defect localization.  
4. The files metrics.py contains the function definition of the metrices used for comparision, which is IOU, recall, precision, accuracy and f1score.  
5. The files resnet.py and segnet.py contains the models for ResNet and SegNet respectively.  
6. The files deeplab.py, load_weights.py and extract_weights.py correspond to loading and extracting weights using DeepLab.  

## Procedure to train the images:  
 The script batch_processing.py can be run with the following command-line arguments:  
   1 : Unet.   
   2 : ResNet.  
   3 : SegNet.  
   Run the script in the following command: python batch_processing.py 1/2/3.  
   
## Procedure to run the predict.py script for predicting test images.  
The predict.py script takes a command-line argument as the folder that contains the images, use the following command to run the script:  
 python predict.py </location/to/folder> <modelname>.  
 Please check that the location of the model is on the same directory for ease of use.
 
 The following verions of libraries were used for the development of this project:  
 Keras:2.2.4  
 TensorFlow:1.4.0  
 Python: 3.6  

## Implementation of this Paper :- http://erk.fe.uni-lj.si/2017/papers/racki(towards_surface).pdf is done in the file fccn.py.   
