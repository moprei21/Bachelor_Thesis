bash environment.sh
python dataset.py --num 8 --h "Dieser Kommentar ist eine persönliche Erfahrung" --outfile 8_engaging.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 8_engaging.csv
python predict.py --name 8_engaging --config engaging

python dataset.py --num 16 --h "Dieser Kommentar ist eine persönliche Erfahrung" --outfile 16_engaging.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 16_engaging.csv
python predict.py --name 16_engaging --config engaging

python dataset.py --num 32 --h "Dieser Kommentar ist eine persönliche Erfahrung" --outfile 32_engaging.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 32_engaging.csv
python predict.py --name 32_engaging --config engaging

python dataset.py --num 64 --h "Dieser Kommentar ist eine persönliche Erfahrung" --outfile 64_engaging.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 64_engaging.csv
python predict.py --name 64_engaging --config engaging

python dataset.py --num 128 --h "Dieser Kommentar ist eine persönliche Erfahrung" --outfile 128_engaging.csv
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 128_engaging.csv
python predict.py --name 128_engaging --config engaging