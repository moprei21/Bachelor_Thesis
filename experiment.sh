bash environment.sh
python dataset.py --num 8 --h "Dieser Kommentar ist eine Beleidigung" --outfile 8_toxic.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y
python predict.py -name 8_toxic