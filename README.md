# Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights (ACM SIGSPATIAL 2019)

## Data
The main data for this work is obtained from [US Accidents](https://smoosavi.org/datasets/us_accidents) and [Large Scale Traffic and Weather Events](https://smoosavi.org/datasets/lstw) datasets. Please visit our paper to learn how to use the raw input datasets and prepare them for our real-time accident prediction framework. Also, several sample data files can be find in ```data``` directory. 

## Generate Input

## Baseline Models
In terms of baselines we employed three models: Logistic Regressions (LR), Gradient Boosted classifier (GBC), and a FeedForward Neural Network Model (DNN). 

## DAP Model
Our Deep Accident Prediction model comprise several important components including _Recurrent Component_, _Embedding Component_, _Description-to-Vector Component_, _Points-Of-Interest Component_, and _Fully-connected Component_. The following image shows a demonstration of this model: <center><img src="/files/D-CRNN_2.png" width="1100"></center>


## How to Run the Code? 
Download the CSV dataset from here: https://smoosavi.org/datasets/us_accidents

Run step1-cleaning-func and step2-acc_prediction_training_preparation in "Step1-preparing for training" folder

Run script in DAB and baseline to train different models. 



## Acknowledgment 
* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. “Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights” In proceedings of the 27th ACM SIGSPATIAL, International Conference on Advances in Geographic Information Systems. ACM, 2019. 

