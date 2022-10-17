bash environment.sh
python dataset.py --num 8 --h " Dieser Kommentar ist eine Beleidigung" --outfile 8_toxic.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 8_toxic.csv
python predict.py --name 8_toxic --config toxic

python dataset.py --num 16 --h " Dieser Kommentar ist eine Beleidigung" --outfile 16_toxic.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 16_toxic.csv
python predict.py --name 16_toxic --config toxic

python dataset.py --num 32 --h " Dieser Kommentar ist eine Beleidigung" --outfile 32_toxic.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 32_toxic.csv
python predict.py --name 32_toxic --config toxic

python dataset.py --num 64 --h " Dieser Kommentar ist eine Beleidigung" --outfile 64_toxic.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 64_toxic.csv
python predict.py --name 64_toxic --config toxic

python dataset.py --num 128 --h " Dieser Kommentar ist eine Beleidigung" --outfile 128_toxic.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 128_toxic.csv
python predict.py --name 128_toxic --config toxic