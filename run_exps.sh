for seed in 0 1 2 3 4 5 6 7 8 9
do
  OMP_NUM_THREADS=1 python main.py --config-file configs/minihack/clear_minihack_small.json --resume-id ${seed}
  OMP_NUM_THREADS=1 python main.py --config-file configs/minihack/impala_minihack_small.json --resume-id ${seed}
done