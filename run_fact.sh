bash environment.sh
python dataset.py --num 8 --h " Dieser Kommentar ist eine externe Quelle" --outfile 8_fact.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 8_fact.csv
python predict.py --name 8_fact --config fact

python dataset.py --num 16 --h " Dieser Kommentar ist eine externe Quelle" --outfile 16_fact.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 16_fact.csv
python predict.py --name 16_fact --config fact

python dataset.py --num 32 --h " Dieser Kommentar ist eine externe Quelle" --outfile 32_fact.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 32_fact.csv
python predict.py --name 32_fact --config fact

python dataset.py --num 64 --h " Dieser Kommentar ist eine externe Quelle" --outfile 64_fact.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 64_fact.csv
python predict.py --name 64_fact --config fact

python dataset.py --num 128 --h " Dieser Kommentar ist eine externe Quelle" --outfile 128_fact.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 128_fact.csv
python predict.py --name 128_fact --config fact