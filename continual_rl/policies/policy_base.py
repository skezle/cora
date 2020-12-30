from abc import ABC, abstractmethod
from continual_rl.experiments.tasks.task_spec import TaskSpec


class PolicyBase(ABC):
    """
    The base class that all agents should implement, enabling them to act in the world.
    """
    def __init__(self):
        """
        Subclass policies will always be initialized with: (config, observation_space, action_spaces).
        No other parameters should be added - the policy won't be loaded with them from the configuration loader.
        Any custom parameters should be put on config.
        observation_space is the common observation space for all tasks
        action_spaces is a map from action_space_id to the action space for a given task.
        """
        pass

    def shutdown(self):
        """
        Indicates the experiment has shutdown, and the policy should cleanup any resources it has open.
        Optional.
        """
        pass

    @abstractmethod
    def get_environment_runner(self, task_spec: TaskSpec):
        """
        Return an instance of the subclass of EnvironmentRunnerBase to be used to run an environment with this policy.
        This is policy-dependent because it determines the cadence and type of observations provided to the policy.
        If the policy supports multiple, which one is used can be configured using the policy_config.
        Each time this function is called, a new EnvironmentRunner should be returned.

        Note: if an environment runner can use multiple environments in parallel and the task_spec indicates we're in
        eval_mode, only one environment should be used. This is because if, say, we want to collect 10 continual eval
        runs, and run 100 envs in parallel to get them, then after some collection we may see only the first 10
        that have returned, which can bias the sample. E.g. if you can die quickly in a game, then the average return
        biased in this manner will be too low.

        :return: an instance of an EnvironmentRunnerBase subclass
        """
        pass

    @abstractmethod
    def compute_action(self, observation, action_space_id, last_timestep_data, eval_mode):
        """
        If a non-synchronous environment runner is specified (or may be in the future), this method should not change
        any instance state, because this method may be run on different  processes or threads to enable parallelization.
        Any information that is needed for updating the policy should be specified in timestep_data.

        :param observation: The expected observation is dependent on what environment runner has been configured for
        the policy, as well as the task type. For instance, an ImageTask with EnvironmentRunnerBatch configured
        will provide an observation that is of shape [batch, time, channels, width, height]. See the documentation for
        collect_data for a given EnvironmentRunner for more detail.
        :param action_space_id: The id of the action space of the task currently being executed. All tasks that use
        the same action space will have the same id.
        :param last_timestep_data: The last timestep_data generated by this policy. Not reset on episode completion.
        (Caller will need to check last_timestep_data.done manually if this is desired.
        :param eval_mode: Boolean indicating whether the policy should be run in eval mode (i.e. not updating)
        :return: (selected actions, timestep_data): timestep_data is an object arbitrarily specified by the subclass.
        It should contain whatever extra information is required for training. A list of lists of timestep_data are
        provided to train(), and are described more there. Actions should always be a list of actions (len=1 for sync)
        """
        pass

    @abstractmethod
    def train(self, storage_buffer):
        """
        By default, training will not be parallelized, therefore this method may freely update instance state.
        :param storage_buffer: A list of lists: [[(timestep_data, reward, done)]]. Each inner list represents the data
        collected by a single process since the last time train() was called. This list is generated by the
        EnvironmentRunner, so further details can be viewed in EnvironmentRunnerBase collect_data.
        :return: None
        """
        pass

    @abstractmethod
    def save(self, output_path_dir, task_id, task_total_steps):
        """
        Saving is delegated to the policy, as there may be more complexity than just torch.save().
        :param output_path_dir: The directory to which the model should be saved
        :param task_id: The task currently being executed when a save was triggered
        :param task_total_steps: The number of steps into this task we are at the time of saving.
        :return: The full path to the saved file
        """
        pass

    @abstractmethod
    def load(self, model_path):
        """
        Load the model from model_path.
        :param model_path: The path 
        :return: The loaded model (can be self if the model was loaded into the current policy)
        """
        pass
