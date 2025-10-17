# 1) Preprocess (parse + engineer features + save arrays)
python preprocessing.py --input MOGE_Coordinates_sin_cos_RT_100pt.txt --outdir "postprocessing_data"

# 2) Train (stratified 70/10/20, StandardScaler on train only, 5-model ensemble)
python fcnn.py --datadir "postprocessing_data" --epochs 1000 --batch_size 16 --n_models 5

# 3) Evaluate + plots (auto-ensembles model_m*.keras if found)
python evaluate.py --datadir "postprocessing_data"
