# Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights (ACM SIGSPATIAL 2019)

## Generate Input
One important process is to transform raw input data into the form of input for a machine learning model. Here we employed multiple processes as follows:

* __Step 1__: Run `1-CreateInputForAccidentPrediction.ipynb` from `/1-GenerateFeatureVector` to generate raw feature vectors. Each vector represents a geographical region of size 5km x 5km (that we call it a geohash) during a 15 minutes time interval. This code uses [LSTW](https://smoosavi.org/datasets/lstw) dataset for traffic events data, raw weather observation records for weather-related attributes (check `data/Sample_Weather.tar.gz` for sample data), and daylight information (check `data/sample_daylight.csv` for sample data). 

* __Step 2__: Run `2-CreateNaturalLanguageRepresentationForGeoHashes.ipynb` to generate description to vector representation for geographical regions. The main inputs for this process are [LSTW](https://smoosavi.org/datasets/lstw) and [GloVe](https://nlp.stanford.edu/projects/glove/). A sample output can be find as `data/geohash_to_text_vec.csv`. 

* __Step 3__: Run `3-DataCleaningAndIntegration.ipynb` for data cleaning, and preparation for integration with POI data. 

* __Step 4__: Run `4-FinalTrainAndTestDataPreparation.ipynb` to prepare final train and test data. This includes creating sample entries, and negative sampling for non-accident data samples. There is two version code: single thread vs multi-thread. The multi-thread version uses more system cores but it needs more memory as well. It is more suitable for running on servers. Single-thread is for running on desktop devices for generating smaller train-test sets. 

Implementations of these steps can be found in `1-GenerateFeatureVector`. Also, note that the sample data and codes are for those cities that we used in the paper (e.g., Atlanta, Austin, Charlotte, Dallas, Houston, and Los Angeles). 

## Sample Data Files For Train and Test
To train and test our proposed model and the baselines, you can use our pre-generated train and test files for six cities Atlanta, Austin, Charlotte, Dallas, Houston, and Los Angeles. The time frame to generate sample data for these cities is the same as what we described in our paper. You can find these files in `data/train_set.7z`. Use `7za -e train_set.7z` to decompress this file and obtain 4 numpy (.npy) files per city. Two files contain feature vectors for train and test, and two files contain train and test labels. These sample files are the result of the above [input generation process](https://github.com/mhsamavatian/DAP/blob/master/README.md#generate-input). 

## Deep Accident Prediction (DAP) Model
Our Deep Accident Prediction model comprises several important components including _Recurrent Component_, _Embedding Component_, _Description-to-Vector Component_, _Points-Of-Interest Component_, and _Fully-connected Component_. The following image shows a demonstration of this model: <center><img src="/files/dap.png" width="600"></center>

The implementation of this model can be found here: `2-DAP/DAP.ipynb`. 

## Baseline Models
In terms of baselines, we employed the following models: 

* Logistic Regressions (LR): Find sample code in `3-Baselines/Traditional_Models_Sklearn.py`. 
* Gradient Boosted classifier (GBC): Find sample code in `3-Baselines/Traditional_Models_Sklearn.py`. 
* FeedForward Neural Network Model (DNN): An implementation of this model can be found in `3-Baselines/DNN.ipynb`. 
* DAP Without Embedding Component (DAP-NoEmbed): An implementation of this model can be found in `2-DAP/DAP-NoEmbed.ipynb`. 

## How to Run the Code? 
All implementations are in `python`, with deep learning models developed in `Keras` using `Tensorflow` as backend. For non-deep learning baselines (i.e., LR and GBC) you can run codes on CPU machines. But for deep-learning models, we recommend using GPU machines to speed-up the process. 

## Acknowledgment 
* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. “Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights” In proceedings of the 27th ACM SIGSPATIAL, International Conference on Advances in Geographic Information Systems. ACM, 2019. 
