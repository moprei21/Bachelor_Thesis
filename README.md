# Bachelor_Thesis
This is the repository of my submission of GermEval21 using Zero and Few-Shot techniques


## Datasets 
To create the datasets run the script create_dataset.sh

## Zero-Shot 
To find the code on Zero-Shot Text Classification use the Notebook germeval_2021.ipynb


## Few-Shot
Run experiments on all subtasks by executing run_toxic.sh, run_engaging.sh, run_fact-sh  

Make sure to have generated all datasets and have an existing Weights and Biases account to visualize the results

### Training the model
use the parameters learning_rate and num_train_epochs to change to different hyperparameter setting in finetune_germeval.py
