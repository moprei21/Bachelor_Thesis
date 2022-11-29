bash environment.sh

python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/fact/8_fact.csv
python predict_fact.py --name 8_fact --threshold 0.1


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename  data/fact/16_fact.csv
python predict_fact.py --name 16_fact --threshold 0.1


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename  data/fact/32_fact.csv
python predict_fact.py --name 32_fact --threshold 0.1


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename  data/fact/64_fact.csv
python predict_fact.py --name 64_fact --threshold 0.1


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename  data/fact/128_fact.csv
python predict_fact.py --name 128_fact --threshold 0.1

python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename  data/fact/256_fact.csv
python predict_fact.py --name 256_fact --threshold 0.1