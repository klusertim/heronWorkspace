# Anomaly Detection for Photo Traps using an Autoencoder
TODO: beschrieb

## Pipeline
### 0 Preprocessing
This section is about preprocessing the data to fit the autoencoder.
#### 0.1 cameraProps
In dataPreprocessing/cameraProps.json, we specify a range to exclude images that would disturb the training of the AE. 

#### 0.2 Classify Data in Grayscale/RGB Images and Motion/Time Images
This is done by the dataPreprocessing/classifyMotionGray.py. This script will generate a ImageData/imageProps\*CAM\*.csv for each specified Camera. To classify the images of camera X and Y, use it the following way: 

```sh
    python dataPreprocessing/classifyMotionGray.py --cameras X Y
```

Additionally it specifies bad images.

#### 0.3 Manual Data Validation Framework
See datasetValidation.ipynb to generate a dataset for testing the pipeline. Running it will first ask for a camera to do the manual classification for and then stores the data in the folder MotionGrayClassification

### 1 CAE
We can specify the model in the src/models.py to use it while training
In trainCAE.py we specify the parameter for the domain search with the number of models that should be trained. Running it will start the training and store the trained models to the specified output path.

### 2 postProcessing / classification
#### performance evaluation
This happens in the basicClassifier.ipynb
