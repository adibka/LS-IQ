from experiment_launcher import Launcher

from experiment_launcher.utils import bool_local_cluster

if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = True
    
    N_SEEDS = 5
    N_EXPS_IN_PARALLEL = 8          # or os.cpu_count() to use all cores
    N_CORES = N_EXPS_IN_PARALLEL
    MEMORY_SINGLE_JOB = 1000
    MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES

    launcher = Launcher(exp_name='lsiq_10',
                        exp_file='lsiq_experiments',
                        n_seeds=N_SEEDS,
                        n_exps_in_parallel=N_EXPS_IN_PARALLEL,
                        n_cores=N_CORES,
                        memory_per_core=MEMORY_PER_CORE,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        )
                        
    default_params = dict(n_epochs=40,
                          n_steps_per_epoch=10000,
                          n_eval_episodes=10,
                          n_steps_per_fit=1,
                          n_epochs_save=-1,
                          logging_iter=10000,
                          gamma=0.99,
                          use_cuda=True,
                          tau=0.005,
                          use_target=True,
                          loss_mode_exp="fix",
                          regularizer_mode="plcy",
                          learnable_alpha=False,
                          num_trajs=10)

    log_std = [(-5, 2)]
    envs = ["Ant-v2",
            "HalfCheetah-v2",
            "Hopper-v2",
            "Humanoid-v2",
            "Walker2d-v2"]
    path_to_datasets = "../../experts/"
    expert_data_filenames = ["Ant-v2_25.npz",
                             "HalfCheetah-v2_25.npz",
                             "Hopper-v2_25.npz",
                             "Humanoid-v2_25.npz",
                             "Walker2d-v2_25.npz"]
    
    expert_data_paths = [path_to_datasets + name for name in expert_data_filenames]

    # Ant
    launcher.add_experiment(env_id__=envs[0], expert_data_path=expert_data_paths[0],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)


    # HalfCheetah
    launcher.add_experiment(env_id__=envs[1], expert_data_path=expert_data_paths[1],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)

    # Hopper
    launcher.add_experiment(env_id__=envs[2], expert_data_path=expert_data_paths[2],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)

    # Humanoid
    launcher.add_experiment(env_id__=envs[3], expert_data_path=expert_data_paths[3],
                            plcy_loss_mode__="value", init_alpha__=0.1, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)

    # Walker2d
    launcher.add_experiment(env_id__=envs[4], expert_data_path=expert_data_paths[4],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)

    launcher.run(LOCAL, TEST)
