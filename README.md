# Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights (ACM SIGSPATIAL 2019)

## Data
The main data for this work is obtained from [US Accidents](https://smoosavi.org/datasets/us_accidents) and [Large Scale Traffic and Weather Events](https://smoosavi.org/datasets/lstw) datasets. Please visit [our paper](https://arxiv.org/pdf/1909.09638.pdf) to learn how to use the raw input datasets and prepare them for our real-time accident prediction framework. Also, several sample data files can be found in ```data``` directory. 

## Generate Input
One important process is to transform raw input data into the form of input for a machine learning model. Here we employed multiple processes as follows:

* __Step 1__: Run `1-CreateInputForAccidentPrediction.ipynb` from `/1-GenerateFeatureVector` to generate raw feature vectors. Each vector represents a geographical region of size 5km x 5km (that we call it a geohash) during a 15 minutes time interval. This code uses [LSTW](https://smoosavi.org/datasets/lstw) dataset for traffic events data, raw weather observation records for weather-related attributes (check `data/Sample_Weather.tar.gz` for sample data), and daylight information (check `data/sample_daylight.csv` for sample data). 

* __Step 2__: Run `2-CreateNaturalLanguageRepresentationForGeoHashes.ipynb` to generate description to vector representation for geographical regions. The main inputs for this process are [LSTW](https://smoosavi.org/datasets/lstw) and [GloVe](https://nlp.stanford.edu/projects/glove/). A sample output can be find as `data/geohash_to_text_vec.csv`. 

* __Step 3__: Run `3-DataCleaningAndIntegration.ipynb` for data cleaning, and preparation for integration with POI data. 

* __Step 4__: Run `4-FinalTrainAndTestDataPreparation.ipynb` to prepare final train and test data. This includes creating sample entries, and negative sampling for non-accident data samples. 

Implementations of these steps can be found in `1-GenerateFeatureVector`. Also, note that the sample data and codes are for those cities that we used in the paper (e.g., Atlanta, Austin, Charlotte, Dallas, Houston, and Los Angeles). 

## Sample Data Files For Train and Test
To train and test our proposed model and the baselines, you can use our pre-generated train and test files for six cities Atlanta, Austin, Charlotte, Dallas, Houston, and Los Angeles. The time frame to generate sample data for these cities is the same as what we described in our paper. You can find these files in `data/train_set.7z`. Use `7za -e train_set.7z` to decompress this file and obtain 4 numpy (.npy) files per city. Two files contain feature vectors for train and test, and two files contain train and test labels. 

## Deep Accident Prediction (DAP) Model
Our Deep Accident Prediction model comprises several important components including _Recurrent Component_, _Embedding Component_, _Description-to-Vector Component_, _Points-Of-Interest Component_, and _Fully-connected Component_. The following image shows a demonstration of this model: <center><img src="/files/dap.png" width="600"></center>

The implementation of this model can be found in the `2-DAP` directory. 

## Baseline Models
In terms of baselines, we employed three models: Logistic Regressions (LR), Gradient Boosted classifier (GBC), and a FeedForward Neural Network Model (DNN). The implementation of these models can be found in `3-Baselines`. There you find one `python` script for each baseline, with instruction on how to run each script provided inside the script. 

## How to Run the Code? 
All implementations are in `python`, with deep learning models developed in `Keras` using `Tensorflow` as backend. For non-deep learning baselines (i.e., LR and GBC) you can run codes on CPU machines. But for deep-learning models, we recommend using GPU machines to speed-up the process. 

Run step1-cleaning-func and step2-acc_prediction_training_preparation in "Step1-preparing for training" folder
Run the script in DAB and baseline to train different models. 

## Acknowledgment 
* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. “Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights” In proceedings of the 27th ACM SIGSPATIAL, International Conference on Advances in Geographic Information Systems. ACM, 2019. 
