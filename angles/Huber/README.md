python preprocessing.py --input MOGE_Coordinates_sin_cos_RT_100pt.txt --outdir "postprocessing_data"
python fcnn.py --datadir "postprocessing_data" --epochs 1000 --batch_size 8 --n_models 7
python evaluate.py --datadir "postprocessing_data"
