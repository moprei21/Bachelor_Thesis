bash environment.sh
python dataset.py --num 8 --outfile 8_engaging.csv --task e
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 8_engaging.csv
python predict_engaging.py --name 8_engaging

python dataset.py --num 16 --outfile 16_engaging.csv --task e
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 16_engaging.csv
python predict_engaging.py --name 16_engaging 

python dataset.py --num 32 --outfile 32_engaging.csv --task e
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 32_engaging.csv
python predict_engaging.py --name 32_engaging 

python dataset.py --num 64 --outfile 64_engaging.csv --task e
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 64_engaging.csv
python predict_engaging.py --name 64_engaging 

python dataset.py --num 128 --outfile 128_engaging.csv --task e
python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename 128_engaging.csv
python predict_engaging.py --name 128_engaging 