# 1) Preprocess (parse + save raw arrays)
python preprocessing.py --input MOGE_Coordinates_sin_cos_RT_100pt.txt --outdir "postprocessing_data"

# 2) Train FCNN (splits, scales, trains, saves artifacts)
python fcnn.py --datadir "postprocessing_data" --epochs 1000 --batch_size 16

# 3) Evaluate and plot
python evaluate.py --datadir "postprocessing_data"
