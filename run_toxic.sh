bash environment.sh

python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/toxic/8_toxic.csv
python predict_toxic.py --name 8_toxic 


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/toxic/16_toxic.csv
python predict_toxic.py --name 16_toxic


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/toxic/32_toxic.csv
python predict_toxic.py --name 32_toxic


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/toxic/64_toxic.csv
python predict_toxic.py --name 64_toxic


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/toxic/128_toxic.csv
python predict_toxic.py --name 128_toxic