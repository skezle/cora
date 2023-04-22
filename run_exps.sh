for seed in 0 1 2 3 4 5 6 7 8 9
do
  OMP_NUM_THREADS=1 python main.py --config-file configs/minigrid/clear_minigrid.json --resume-id ${seed}
  #OMP_NUM_THREADS=1 python main.py --config-file configs/minigrid/clear_minigrid_smallrp.json --resume-id ${seed}
  #OMP_NUM_THREADS=1 python main.py --config-file configs/minigrid/clear_minigrid_tinyrp.json --resume-id ${seed}
done
