# Anomaly detection pipeline for images from low-framerate camera traps

This is the repository containing the code for the anomaly detection pipeline.
Please consider reading the paper for a full explanation of the concepts behind the pipeline.

The pipeline was developed on a Swiss Data Science Center (SDSC) computer that contains the photo trap data (the data is not in the repo). Therefore, several scripts only work with the file structure of SDSC's computer. 

The pipeline is classified into three sections (three folders):
1. 0_preProcessing
2. 1_AE
3. 2_postProcessing

## Pipeline
### 0 Preprocessing
This section is about extracting metadata to use it in the previous sections.

#### 0.1 cameraProps
In dataPreprocessing/cameraProps.json, we can specify multiple ranges to exclude images that would disturb the training of the AE. The JSON has the following structure: 
```json
"camera1": {
        "exclude": {
            [
            [imageNameStart1, imageNameStop1],
            [imageNameStart2, imageNameStop2],
            ...]}
    },
    "camera2": {
    ...
    }
```

#### 0.2 Classify Data in Grayscale/RGB Images and Motion/Time Images
The 0_preProcessing/classifyMotionGray.py does this. This script will generate a MotionGrayClassification/classifiedMotionGray\*CAM\*.csv for each specified camera. To classify the images of cameras X and Y, use the following way: 

```sh
    python 0_preProcessing/classifyMotionGray.py --cameras X Y
```

The CSV contains an entry for each image, specifying if it's grayscale/RGB, captured by motion/time sensor, and if the image is corrupt.

#### 0.3 Manual Data Validation Framework
See datasetValidation.ipynb to generate a dataset for testing the pipeline. Running the first cells will first ask for a camera name to enter. Then, the program samples random images from the camera to classify them into six classes: 
- Pictures containing no anomaly (Class 0)
- Pictures containing a heron that should be easy to see (Class 1)
- Pictures containing a heron that is hard to see (Class 2)
- Pictures containing an anomaly (without herons) that is easy to see (Class 3)
- Pictures containing an anomaly (without herons) that is hard to see (Class 4)
- Corrupt images/Images to exclude from our validated set (Class -1)

The classifications for camera CAMX are stored in manuallyClassified/manuallyClassifiedCAMX.csv.

### 1 CAE

We can specify the model architecture in the 1_AE/models.py and use it in the
1_AE/trainCAE.py. In trainCAE.py, we specify the domain/grid search parameters and the number of models that should be trained. Running trainCAE.py will generate an instance of CAEHeron in AEHeronModel.py, a LightningModule (pytorch lightning). The lightning module handles the training and testing of the autoencoder and stores the checkpoints to the specified output path.
Training the CAE for specific cameras requires each camera's motion/gray classification.

### 2 postProcessing/classification
#### ClassifierDatasets
ClassifierDatasets.py contains the DatasetThreeConstecutive class, which aims to generate useful data to fit the post-processing pipeline and test the entire pipeline. It generates labeled features that consist of three consecutive images. There are several parameters that can be specified:
- set: split the data into train, val, test set, or get all the images
- lblValidatoinMode: Choose one of TinaDubach, MotionSensor, manual validation methods to generate the labels for images.
    - anomalyObviousness: If manual validation was chosen, we can further distinct between obvious (Class 2, 4, 0) and non-obvious (Class 1, 3, 0) images or return all the images (Class 1, 2, 3, 4, 0)
- cameras: list of cameras to use
- balanced: If true, the dataset will be balanced between positive and negative frames on a camera basis.
- distinctCAETraining: If true, the dataset will be distinct from the one used for training the CAE and just take the images classified as captured by the motion sensor
- colorMode: RGB, grayscale, mix
- resize_to: (h, w) to resize the images to
- random_state: random seed for reproducibility
- transform: torch-vision transform to apply to the images - None is the default transform

#### PostProcessingHelper
The PostProcessingHelper.py contains helpful modules and methods for using and evaluating the entire pipeline. 
- Class CheckPoints contains several interesting AE checkpoints
The class MinFilter implements a min-filter for variable kernel size that is used in the post-processing pipeline.
- Method PostProcess::computeSum contains the entire pipeline. The anomaly rating is computed for each image for a given AE-checkpoint, given loaderParams, and other pipeline-specific params.

#### postProcessingAndEval
This folder contains several notebooks for evaluating the performance of the entire pipeline and its parts. 

## Other Folders/Files
### Folder src
This folder contains scripts used in several sections of the pipeline. Currently, only the HeronImageLoader is used in the pre-processing and AE sections.

### Folder thesis
This folder contains some evaluation and generated graphics used for the written part of the thesis.

### Folder logs and lightning_logs
This folder contains several checkpoints for trained models.

### Folder nonUsedCode
This contains dead code that is not further used in our project, such as the code for an MLP or other computer vision approaches.

### Folder interestingData
It contains some additional interesting images and graphics

### condaEnvPackageList
This list defines all the packages of the conda environment we used for this work.