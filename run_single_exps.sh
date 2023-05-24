for seed in {0..19}
do
    ## DoorKey
    OMP_NUM_THREADS=1 python main.py --wandb_group clear_single \
     --wandb_proj_name minigrid_baselines --tag= mg_clear_dk \
     --config-file configs/minigrid/clear_minigrid_doorkey.json --resume-id ${seed}
    OMP_NUM_THREADS=1 python main.py --wandb_group impala_single \
     --wandb_proj_name minigrid_baselines --tag= mg_impala_dk \
    --config-file configs/minigrid/impala_minigrid_doorkey.json --resume-id ${seed}
    
    ## LavaCrossing
    # OMP_NUM_THREADS=1 python main.py --wandb_group clear_single \
    #  --wandb_proj_name minigrid_baselines --tag= mg_clear_lc \
    #  --config-file configs/minigrid/clear_minigrid_lavacrossing.json --resume-id ${seed}
    # OMP_NUM_THREADS=1 python main.py --wandb_group impala_single \
    #  --wandb_proj_name minigrid_baselines --tag= mg_impala_lc \
    # --config-file configs/minigrid/impala_minigrid_lavacrossing.json --resume-id ${seed}

    ## SimpleCrossing
    # OMP_NUM_THREADS=1 python main.py --wandb_group clear_single \
    #  --wandb_proj_name minigrid_baselines --tag= mg_clear_sc \
    #  --config-file configs/minigrid/clear_minigrid_simplecrossing.json --resume-id ${seed}
    # OMP_NUM_THREADS=1 python main.py --wandb_group impala_single \
    #  --wandb_proj_name minigrid_baselines --tag= mg_impala_sc \
    # --config-file configs/minigrid/impala_minigrid_simplecrossing.json --resume-id ${seed}
done


