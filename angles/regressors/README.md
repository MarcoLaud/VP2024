# 1) Preprocess (same as before)
python preprocessing.py --input MOGE_Coordinates_sin_cos_RT_100pt.txt --outdir "postprocessing_data"

# 2) Train + model selection (classical regressors)
python train_regressors.py --datadir "postprocessing_data"

# 3) Evaluate + plots
python evaluate_regressor.py --datadir "postprocessing_data"
