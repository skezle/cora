for seed in {0..19}
do

  ## Minigrid
  OMP_NUM_THREADS=1 python main.py --wandb_group clear --wandb_proj_name minigrid_baselines 
  --tag= mg_clear --config-file configs/minigrid/clear_minigrid.json --resume-id ${seed}
  OMP_NUM_THREADS=1 python main.py --wandb_group impala --wandb_proj_name minigrid_baselines 
  --tag= mg_impala --config-file configs/minigrid/impala_minigrid.json --resume-id ${seed}

  ## 4 task Minihack
  # OMP_NUM_THREADS=1 python main.py --wandb_group clear --wandb_proj_name 4task_baselines \
  # --tag= mg_clear --config-file configs/minigrid/clear_minihack.json --resume-id ${seed}
  # OMP_NUM_THREADS=1 python main.py --wandb_group impala --wandb_proj_name 4task_baselines \
  # --tag= mg_impala --config-file configs/minigrid/impala_minihack.json --resume-id ${seed}
  
  ## 8 task Minihack
  # OMP_NUM_THREADS=1 python main.py --wandb_group clear --wandb_proj_name 8task_baselines \
  # --tag= mg_clear --config-file configs/minigrid/clear_minihack_8task.json --resume-id ${seed}
  # OMP_NUM_THREADS=1 python main.py --wandb_group impala --wandb_proj_name 8task_baselines \
  # --tag= mg_impala --config-file configs/minigrid/impala_clear_8task.json --resume-id ${seed}

done


