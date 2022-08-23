for seed in 0 1 2 3 4
do
  OMP_NUM_THREADS=1 python main.py --config-file configs/minihack/clear_minihack_single.json --resume-id ${seed}
  OMP_NUM_THREADS=1 python main.py --config-file configs/minihack/impala_minihack_single.json --resume-id ${seed}
done