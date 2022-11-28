# Bachelor_Thesis
This is the repository of my submission of GermEval21 using Zero and Few-Shot techniques


## Datasets 
To create the datasets run the script create_dataset.sh

## Zero-Shot 
To find the code on Zero-Shot Text Classification use the Notebook germeval_2021.ipynb


train file command:
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y
