bash environment.sh

python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/engaging/8_engaging.csv
python predict_engaging.py --name 8_engaging


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/engaging/16_engaging.csv
python predict_engaging.py --name 16_engaging



python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/engaging/32_engaging.csv
python predict_engaging.py --name 32_engaging


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/engaging/64_engaging.csv
python predict_engaging.py --name 64_engaging


python finetune_germeval.py --model_name_or_path Sahajtomar/German_Zeroshot --output_dir pretrain_out --language de --do_train y --overwrite_output_dir y --filename data/engaging/128_engaging.csv
python predict_engaging.py --name 128_engaging 