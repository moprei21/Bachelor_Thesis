cd data
mkdir toxic
mkdir engaging
mkdir fact

cd ..

python3 dataset.py --num 8 --outfile data/toxic/8_toxic.csv --task t
python3 dataset.py --num 16 --outfile data/toxic/16_toxic.csv --task t
python3 dataset.py --num 32 --outfile data/toxic/32_toxic.csv --task t
python3 dataset.py --num 64 --outfile data/toxic/64_toxic.csv --task t
python3 dataset.py --num 128 --outfile data/toxic/128_toxic.csv --task t




python3 dataset.py --num 8 --outfile data/engaging/8_engaging.csv --task e
python3 dataset.py --num 16 --outfile data/engaging/16_engaging.csv --task e
python3 dataset.py --num 32 --outfile data/engaging/32_engaging.csv --task e
python3 dataset.py --num 64 --outfile data/engaging/64_engaging.csv --task e
python3 dataset.py --num 128 --outfile data/engaging/128_engaging.csv --task e

python3 dataset.py --num 8 --outfile data/fact/8_fact.csv --task f
python3 dataset.py --num 16 --outfile data/fact/16_fact.csv --task f
python3 dataset.py --num 32 --outfile data/fact/32_fact.csv --task f
python3 dataset.py --num 64 --outfile data/fact/64_fact.csv --task f
python3 dataset.py --num 128 --outfile data/fact/128_fact.csv --task f


