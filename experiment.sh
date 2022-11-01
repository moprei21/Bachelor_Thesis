bash environment.sh
python dataset.py --num 8 --outfile 8_toxic.csv --task t
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 8_toxic.csv
python predict_toxic.py --name 8_toxic

python dataset.py --num 16 --outfile 16_toxic.csv --task t
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 16_toxic.csv
python predict_toxic.py --name 16_toxic

python dataset.py --num 32 --outfile 32_toxic.csv --task t
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 32_toxic.csv
python predict_toxic.py --name 32_toxic

python dataset.py --num 64 --outfile 64_toxic.csv --task t
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 64_toxic.csv
python predict_toxic.py --name 64_toxic

python dataset.py --num 128 --outfile 128_toxic.csv --task t
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 128_toxic.csv
python predict_toxic.py --name 128_toxic