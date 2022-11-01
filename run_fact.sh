bash environment.sh
python dataset.py --num 8  --outfile 8_fact.csv --task f
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 8_fact.csv
python predict_fact.py --name 8_fact 

python dataset.py --num 16  --outfile 16_fact.csv --task f
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 16_fact.csv
python predict_fact.py --name 16_fact 

python dataset.py --num 32  --outfile 32_fact.csv --task f
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 32_fact.csv
python predict_fact.py --name 32_fact 

python dataset.py --num 64  --outfile 64_fact.csv --task f
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 64_fact.csv
python predict_fact.py --name 64_fact 

python dataset.py --num 128  --outfile 128_fact.csv --task f
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 128_fact.csv
python predict_fact.py --name 128_fact 