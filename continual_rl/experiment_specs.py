import os.path
import json

from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.experiment_skillhack import ExperimentSkills
#from continual_rl.experiments.tasks.make_atari_task import get_single_atari_task
from continual_rl.experiments.tasks.make_procgen_task import get_single_procgen_task
from continual_rl.experiments.tasks.make_chores_task import create_chores_tasks_from_sequence
from continual_rl.experiments.tasks.make_minihack_task import get_single_minihack_task
#from continual_rl.experiments.tasks.make_skillhack_task import get_single_skillhack_task
from continual_rl.experiments.tasks.minigrid_task import get_single_minigrid_task
from continual_rl.available_policies import LazyDict


# def create_atari_sequence_loader(
#     task_prefix,
#     game_names,
#     num_timesteps=5e7,
#     max_episode_steps=None,
#     full_action_space=False,
#     continual_testing_freq=1000,
#     cycle_count=1,
# ):
#     def loader():
#         tasks = [
#             get_single_atari_task(
#                 f"{task_prefix}_{action_space_id}",
#                 action_space_id,
#                 name,
#                 num_timesteps,
#                 max_episode_steps=max_episode_steps,
#                 full_action_space=full_action_space
#             ) for action_space_id, name in enumerate(game_names)
#         ]
#
#         return Experiment(
#             tasks,
#             continual_testing_freq=continual_testing_freq,
#             cycle_count=cycle_count,
#         )
#     return loader
#
#
# def create_atari_single_game_loader(env_name):
#     return lambda: Experiment(tasks=[
#         # Use the env name as the task_id if it's a 1:1 mapping between env and task (as "single game" implies)
#         get_single_atari_task(env_name, 0, env_name, num_timesteps=5e7, max_episode_steps=10000)
#     ])


def create_procgen_sequence_loader(
    task_prefix,
    game_names,
    num_timesteps=5e6,
    task_params=None,
    add_eval_task=True,
    eval_task_override_params=None,
    continual_testing_freq=1000,
    cycle_count=1,
):
    task_params = task_params if task_params is not None else {}
    eval_task_override_params = eval_task_override_params if eval_task_override_params is not None else {}

    def loader():
        tasks = []
        for action_space_id, name in enumerate(game_names):
            task = get_single_procgen_task(
                f"{task_prefix}_{action_space_id}",
                action_space_id,
                name,
                num_timesteps,
                **task_params,
            )
            tasks.append(task)

            if add_eval_task:
                eval_task = get_single_procgen_task(
                    f"{task_prefix}_{action_space_id}_eval",
                    action_space_id,
                    name,
                    0,  # not training with this task
                    eval_mode=True,
                    **{**task_params, **eval_task_override_params}  # order matters, overriding params
                )
                tasks.append(eval_task)

        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    return loader


def create_chores_sequence_loader(
    task_prefix,
    cycle_count=1,
    continual_testing_freq=5e4,
    num_timesteps=2e6,
    max_episode_steps=1000,
    sequence_file_name="alfred_task_sequences.json"
):
    def loader():
        tasks = create_chores_tasks_from_sequence(task_prefix, sequence_file_name, num_timesteps, max_episode_steps)
        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    return loader


def create_minihack_loader(
    task_prefix,
    env_names,
    num_timesteps=10e6,
    task_params=None,
    continual_testing_freq=1000,
    cycle_count=1,
):
    task_params = task_params if task_params is not None else {}

    def loader():
        tasks = []
        for action_space_id, env_name in enumerate(env_names):
            task = get_single_minihack_task(
                f"{task_prefix}_{action_space_id}",
                action_space_id, 
                env_name,
                num_timesteps,
                **task_params
            )
            # eval_task = get_single_minihack_task(f"{task_prefix}_{action_space_id}_eval", action_space_id, pairs[1],
            #                                      0, eval_mode=True, **task_params)

            tasks += [task]

        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    return loader

def create_skillhack_loader(
    task_prefix,
    skill_names,
    task_names,
    skill_num_timesteps=10e6,
    task_num_timesteps=50e6,
    skill_params=None,
    task_params=None,
    continual_testing_freq=1000,
    cycle_count=1,
):
    skill_params = skill_params if skill_params is not None else {}
    task_params = task_params if task_params is not None else {}

    flags = {
        "save_tty": False,
        "obs_keys": "glyphs,chars,colors,specials,blstats,message",
        "penalty_step": -0.001,
        "penalty_time": 0.0,
        "fn_penalty_step": "constant",
        "max_num_steps": 1000,
        "character": None,
        "state_counter": "none",
        "seedspath": "",
    }

    def loader():
        skills = []
        for action_space_id, env_name in enumerate(skill_names):
            skill = get_single_skillhack_task(
                f"{task_prefix}_{action_space_id}", 
                action_space_id,
                env_name,
                flags,
                skill_num_timesteps, 
                **skill_params,
            )

            skills += [skill]

        tasks = []
        for action_space_id, env_name in enumerate(task_names):
            task = get_single_skillhack_task(
                f"{task_prefix}_{action_space_id}", 
                action_space_id,
                env_name,
                flags,
                task_num_timesteps, 
                **task_params,
            )

            tasks += [task]

        return ExperimentSkills(
            skills,        
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    
    return loader


def create_minigrid_loader(
    task_prefix,
    env_name_pairs,
    num_timesteps=1e6,
    task_params=None,
    continual_testing_freq=1000,
    cycle_count=1,
    config='env_config.json',
):
    task_params = task_params if task_params is not None else {}

    with open(os.path.join(os.path.dirname(__file__),
                           '../configs/minigrid',
                           config), 'r') as f:
        params = json.load(f)
        print(params)
        env_seed = params['env_seed']

    def loader():
        tasks = []
        for action_space_id, pairs in enumerate(env_name_pairs):
            train_task = get_single_minigrid_task(f"{task_prefix}_{action_space_id}", action_space_id, pairs[0],
                                                  num_timesteps, env_seed=env_seed, **task_params)
            eval_task = get_single_minigrid_task(f"{task_prefix}_{action_space_id}_eval", action_space_id, pairs[1],
                                                 0, eval_mode=True, env_seed=env_seed)

            tasks += [train_task, eval_task]

        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )

    return loader

def get_available_experiments():

    experiments = LazyDict({

        # ===============================
        # ============ Atari ============
        # ===============================

        # "adventure": create_atari_single_game_loader("AdventureNoFrameskip-v4"),
        # "air_raid": create_atari_single_game_loader("AirRaidNoFrameskip-v4"),
        # "alien": create_atari_single_game_loader("AlienNoFrameskip-v4"),
        # "amidar": create_atari_single_game_loader("AmidarNoFrameskip-v4"),
        # "assault": create_atari_single_game_loader("AssaultNoFrameskip-v4"),
        # "asterix": create_atari_single_game_loader("AsterixNoFrameskip-v4"),
        # "asteroids": create_atari_single_game_loader("AsteroidsNoFrameskip-v4"),
        # "atlantis": create_atari_single_game_loader("AtlantisNoFrameskip-v4"),
        # "bank_heist": create_atari_single_game_loader("BankHeistNoFrameskip-v4"),
        # "battle_zone": create_atari_single_game_loader("BattleZoneNoFrameskip-v4"),
        # "beam_rider": create_atari_single_game_loader("BeamRiderNoFrameskip-v4"),
        # "berzerk": create_atari_single_game_loader("BerzerkNoFrameskip-v4"),
        # "bowling": create_atari_single_game_loader("BowlingNoFrameskip-v4"),
        # "boxing": create_atari_single_game_loader("BoxingNoFrameskip-v4"),
        # "breakout": create_atari_single_game_loader("BreakoutNoFrameskip-v4"),
        # "carnival": create_atari_single_game_loader("CarnivalNoFrameskip-v4"),
        # "centipede": create_atari_single_game_loader("CentipedeNoFrameskip-v4"),
        # "chopper_command": create_atari_single_game_loader("ChopperCommandNoFrameskip-v4"),
        # "crazy_climber": create_atari_single_game_loader("CrazyClimberNoFrameskip-v4"),
        # "demon_attack": create_atari_single_game_loader("DemonAttackNoFrameskip-v4"),
        # "double_dunk": create_atari_single_game_loader("DoubleDunkNoFrameskip-v4"),
        # "elevator_action": create_atari_single_game_loader("ElevatorActionNoFrameskip-v4"),
        # "fishing_derby": create_atari_single_game_loader("FishingDerbyNoFrameskip-v4"),
        # "frostbite": create_atari_single_game_loader("FrostbiteNoFrameskip-v4"),
        # "gopher": create_atari_single_game_loader("GopherNoFrameskip-v4"),
        # "gravitar": create_atari_single_game_loader("GravitarNoFrameskip-v4"),
        # "hero": create_atari_single_game_loader("HeroNoFrameskip-v4"),
        # "ice_hockey": create_atari_single_game_loader("IceHockeyNoFrameskip-v4"),
        # "james_bond": create_atari_single_game_loader("JamesbondNoFrameskip-v4"),
        # "journey_escape": create_atari_single_game_loader("JourneyEscapeNoFrameskip-v4"),
        # "kangaroo": create_atari_single_game_loader("KangarooNoFrameskip-v4"),
        # "krull": create_atari_single_game_loader("KrullNoFrameskip-v4"),
        # "kung_fu_master": create_atari_single_game_loader("KungFuMasterNoFrameskip-v4"),
        # "montezuma_revenge": create_atari_single_game_loader("MontezumaRevengeNoFrameskip-v4"),
        # "ms_pacman": create_atari_single_game_loader("MsPacmanNoFrameskip-v4"),
        # "name_this_game": create_atari_single_game_loader("NameThisGameNoFrameskip-v4"),
        # "phoenix": create_atari_single_game_loader("PhoenixNoFrameskip-v4"),
        # "pitfall": create_atari_single_game_loader("PitfallNoFrameskip-v4"),
        # "pong": create_atari_single_game_loader("PongNoFrameskip-v4"),
        # "pooyan": create_atari_single_game_loader("PooyanNoFrameskip-v4"),
        # "private_eye": create_atari_single_game_loader("PrivateEyeNoFrameskip-v4"),
        # "qbert": create_atari_single_game_loader("QbertNoFrameskip-v4"),
        # "riverraid": create_atari_single_game_loader("RiverraidNoFrameskip-v4"),
        # "road_runner": create_atari_single_game_loader("RoadRunnerNoFrameskip-v4"),
        # "robotank": create_atari_single_game_loader("RobotankNoFrameskip-v4"),
        # "seaquest": create_atari_single_game_loader("SeaquestNoFrameskip-v4"),
        # "space_invaders": create_atari_single_game_loader("SpaceInvadersNoFrameskip-v4"),
        # "star_gunner": create_atari_single_game_loader("StarGunnerNoFrameskip-v4"),
        # "tennis": create_atari_single_game_loader("TennisNoFrameskip-v4"),
        # "time_pilot": create_atari_single_game_loader("TimePilotNoFrameskip-v4"),
        # "tutankham": create_atari_single_game_loader("TutankhamNoFrameskip-v4"),
        # "up_n_down": create_atari_single_game_loader("UpNDownNoFrameskip-v4"),
        # "video_pinball": create_atari_single_game_loader("VideoPinballNoFrameskip-v4"),
        # "wizard_of_wor": create_atari_single_game_loader("WizardOfWorNoFrameskip-v4"),
        # "yars_revenge": create_atari_single_game_loader("YarsRevengeNoFrameskip-v4"),
        # "zaxxon": create_atari_single_game_loader("ZaxxonNoFrameskip-v4"),
        #
        # # # Default action space for games in used in atari_cycle
        # # {
        # #     0: 'SpaceInvadersNoFrameskip-v4',
        # #     1: 'KrullNoFrameskip-v4',
        # #     2: 'BeamRiderNoFrameskip-v4',
        # #     3: 'HeroNoFrameskip-v4',
        # #     4: 'StarGunnerNoFrameskip-v4',
        # #     5: 'MsPacmanNoFrameskip-v4'
        # # }
        # # {0: Discrete(6), 1: Discrete(18), 2: Discrete(9), 3: Discrete(18), 4: Discrete(18), 5: Discrete(9)}
        #
        # "atari_6_tasks_5_cycles": create_atari_sequence_loader(
        #     "atari_6_tasks_5_cycles",
        #     ["SpaceInvadersNoFrameskip-v4",
        #      "KrullNoFrameskip-v4",
        #      "BeamRiderNoFrameskip-v4",
        #      "HeroNoFrameskip-v4",
        #      "StarGunnerNoFrameskip-v4",
        #      "MsPacmanNoFrameskip-v4"],
        #     max_episode_steps=10000,
        #     num_timesteps=5e7,
        #     full_action_space=True,
        #     continual_testing_freq=0.25e6,
        #     cycle_count=5,
        #  ),
        #
        # "mini_atari_3_tasks_3_cycles": create_atari_sequence_loader(
        #     "mini_atari_3_tasks_3_cycles",
        #     ["SpaceInvadersNoFrameskip-v4",
        #      "BeamRiderNoFrameskip-v4",
        #      "MsPacmanNoFrameskip-v4"],
        #     max_episode_steps=10000,
        #     num_timesteps=5e7,
        #     full_action_space=True,
        #     continual_testing_freq=0.25e6,
        #     cycle_count=3,
        # ),

        # ===============================
        # ============ Procgen ==========
        # ===============================

        "procgen_6_tasks_1_cycles": create_procgen_sequence_loader(
            # using same games as section 5.3 of https://openreview.net/pdf?id=Qun8fv4qSby
            "procgen_6_tasks_1_cycles",
            ["climber-v0",
             "dodgeball-v0",
             "ninja-v0",
             "starpilot-v0",
             "bigfish-v0",
             "fruitbot-v0"],
            num_timesteps=5e6, # 25M steps total per environment
            task_params=dict(
                num_levels=200,
                start_level=0,
                distribution_mode="easy",
            ),
            add_eval_task=True,
            eval_task_override_params=dict(
                num_levels=0,  # full distribution
            ),
            continual_testing_freq=0.25e6,
            cycle_count=1,
        ),

        # ===============================
        # ============ MiniHack =========
        # ===============================
        "minihack_nav_paired_2_cycles": create_minihack_loader(
            "minihack_nav_paired_2_cycles",
            [
                ("Room-Random-5x5-v0", "Room-Random-15x15-v0"),
                ("Room-Dark-5x5-v0", "Room-Dark-15x15-v0"),
                ("Room-Monster-5x5-v0", "Room-Monster-15x15-v0"),
                ("Room-Trap-5x5-v0", "Room-Trap-15x15-v0"),
                ("Room-Ultimate-5x5-v0", "Room-Ultimate-15x15-v0"),
                ("Corridor-R2-v0", "Corridor-R5-v0"),
                ("Corridor-R3-v0", "Corridor-R5-v0"),
                ("KeyRoom-S5-v0", "KeyRoom-S15-v0"),
                ("KeyRoom-Dark-S5-v0", "KeyRoom-Dark-S15-v0"),
                ("River-Narrow-v0", "River-v0"),
                ("River-Monster-v0", "River-MonsterLava-v0"),
                ("River-Lava-v0", "River-MonsterLava-v0"),
                ("HideNSeek-v0", "HideNSeek-Big-v0"),
                ("HideNSeek-Lava-v0", "HideNSeek-Big-v0"),
                ("CorridorBattle-v0", "CorridorBattle-Dark-v0")
            ],
            num_timesteps=10e6,
            continual_testing_freq=1e6,
            cycle_count=2,
        ),

        "minihack_nav_1_cycles": create_minihack_loader(
            "minihack_nav_1_cycles",
            [
                "Room-Random-15x15-v0",  # |A|=8 consider replacing with "Room-Ultimate-5x5-v0",
                "Room-Monster-15x15-v0",  # |A|=8
                "Room-Trap-15x15-v0",  # |A|=8
                "Room-Ultimate-15x15-v0",  # |A|=8
                "River-Narrow-v0",
                "River-v0",
                "River-Monster-v0",
                # "CorridorBattle-v0",
                "HideNSeek-v0",
                # "Room-Random-5x5-v0",
                # "Room-Monster-5x5-v0",
                # "Room-Trap-5x5-v0",
                # # "Corridor-R2-v0",
                # # "KeyRoom-S5-v0",
                # "River-Narrow-v0",
                # "River-Monster-v0",
                # "CorridorBattle-v0",
                # "River-Lava-v0",
                # "River-MonsterLava-v0"
            ],
            num_timesteps=1e6,
            continual_testing_freq=1e4,
            cycle_count=1,
        ),

        "minihack_small": create_minihack_loader(
            "minihack_small",
            [
                "Room-Random-15x15-v0",  # |A|=8 consider replacing with "Room-Ultimate-5x5-v0",
                "Room-Trap-15x15-v0",  # |A|=8
                "River-Narrow-v0",
                "River-Monster-v0",
            ],
            num_timesteps=1e6,
            continual_testing_freq=1e4,
            cycle_count=1,
        ),

        "minihack_single": create_minihack_loader(
            "minihack_single",
            [
                "Room-Random-15x15-v0",
                #"Room-Monster-15x15-v0",
                #"Room-Trap-15x15-v0",
                #"Room-Ultimate-15x15-v0",
                #"River-Narrow-v0",
                #"River-v0",
                #"River-Monster-v0",
                #"HideNSeek-v0",
            ],
            num_timesteps=1e6,
            continual_testing_freq=1e4,
            cycle_count=1,
        ),

        # ===============================
        # ============ MiniGrid =========
        # ===============================
        # the pairs of envs are (train_env, test_env)
        # so in TB on the odd envs are the test envs which we
        # can grab the episodic rewards to report.
        "minigrid_1_cycles": create_minigrid_loader(
            "minigrid_1_cycles",
            [
                ('MiniGrid-DoorKey-9x9-v0', 'MiniGrid-DoorKey-9x9-v0'),
                ('MiniGrid-LavaCrossingS9N1-v0', 'MiniGrid-LavaCrossingS9N1-v0'),
                ('MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-SimpleCrossingS9N1-v0'),
            ],
            num_timesteps=7.5e5,
            continual_testing_freq=1e4,
            cycle_count=1,
            config='env_config.json',
        ),

        "minigrid_single_doorkey": create_minigrid_loader(
            "minihack_single",
            [
                ('MiniGrid-DoorKey-9x9-v0', 'MiniGrid-DoorKey-9x9-v0'),
            ],
            num_timesteps=7.5e5,
            continual_testing_freq=1e4,
            cycle_count=1,
            config='env_config.json',
        ),
        "minigrid_single_lavacrossing": create_minigrid_loader(
            "minihack_single",
            [
                ('MiniGrid-LavaCrossingS9N1-v0', 'MiniGrid-LavaCrossingS9N1-v0'),
            ],
            num_timesteps=7.5e5,
            continual_testing_freq=1e4,
            cycle_count=1,
            config='env_config.json',
        ),
        "minigrid_single_simplecrossing": create_minigrid_loader(
            "minihack_single",
            [
                ('MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-SimpleCrossingS9N1-v0'),
            ],
            num_timesteps=7.5e5,
            continual_testing_freq=1e4,
            cycle_count=1,
            config='env_config.json',
        ),

        # ==================================
        # ============ SkillHack ===========
        # ==================================
        "skillhack_small": create_skillhack_loader(
            "skillhack_small",
            [   
                #"mini_skill_apply_frost_horn",
                "mini_skill_eat"
                "mini_skill_fight"
                "mini_skill_nav_blind",
                #"mini_skill_nav_lava",
                #"mini_skill_nav_lava_to_amulet",
                "mini_skill_nav_water",
                "mini_skill_pick_up",   
                "mini_skill_put_on",
                "mini_skill_take_off",
                #"mini_skill_throw",
                #"mini_skill_unlock",
                "mini_skill_wear",
                "mini_skill_wield",
                #"mini_skill_zap_cold",
                #"mini_skill_zap_death",
            ],
            [
                "mini_simple_seq", # : Battle
                # "mini_simple_union", # : Over or Around
                "mini_simple_intersection", # : Prepare for Battle
                # "mini_simple_random", # : Target Practice
                # "mini_lc_freeze", #: Frozen Lava Cross
                "mini_medusa", # : Medusa
                # "mini_mimic", # : Identify Mimic
                "mini_seamonsters", # : Sea Monsters
            ],
            # skill_num_timesteps=1e7,
            skill_num_timesteps=1e6,
            #task_num_timesteps=50e6,
            task_num_timesteps=1e6,
            continual_testing_freq=1e4,
            cycle_count=1,
        ),

        # ===============================
        # ============ CHORES ===========
        # ===============================

        # Verified set, using replay_checks
        "chores_vary_objects_sequential_clean": create_chores_sequence_loader(
            "chores_vary_objects_sequential_clean",
            num_timesteps=1e6,
            max_episode_steps=1000,
            sequence_file_name='chores/vary_objects.json',
            cycle_count=2),
        "chores_vary_tasks_sequential_tp": create_chores_sequence_loader(
            "chores_vary_tasks_sequential_tp",
            num_timesteps=1e6,
            max_episode_steps=1000,
            sequence_file_name='chores/vary_tasks.json',
            cycle_count=2),
        "chores_vary_envs_sequential_pick_handtowel": create_chores_sequence_loader(
            "chores_vary_envs_sequential_pick_handtowel",
            num_timesteps=1e6,
            max_episode_steps=1000,
            sequence_file_name='chores/vary_envs.json',
            cycle_count=2),
        "chores_sequential_multi_traj": create_chores_sequence_loader(
            "chores_sequential_multi_traj",
            num_timesteps=1e6,
            max_episode_steps=1000,
            sequence_file_name='chores/multi_traj.json',
            cycle_count=2),

    })

    return experiments
