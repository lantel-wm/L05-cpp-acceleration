from assmanager import AssManager

inflation_values = [1.0, 1.01]
inflation_sequences = ['before_DA']
ensemble_size = 40
forcings = [16, 15]
time_steps = 200 * 360 * 5
# time_steps = 200 * 20
configs = []

for inf in inflation_values:
    for seq in inflation_sequences:
        for forcing in forcings:
            configs.append(
                {
                    'model_params': {
                        'forcing': forcing,
                        'time_steps': time_steps,
                    },
                    
                    'DA_params': {
                        'time_steps': time_steps,
                    },
                    
                    'DA_config': {
                        'ensemble_size': ensemble_size,
                        'inflation_factor': inf,
                        'inflation_sequence': seq,
                        'filter': 'EnKF',
                    },
                    
                    'DA_option': {
                        'save_kalman_gain': True,
                        'save_prior_ensemble': True,
                        'save_analysis_ensemble': True,
                        'file_save_option': 'multiple_files',
                    },

                    'Experiment_option': {
                        'experiment_name': f'EnKF_F{forcing}_inf_{inf}_{seq}_sz{ensemble_size}_5y_cpptest',
                        'result_save_path': '/mnt/pve_nfs/zyzhao/L05_experiments',
                    }
                }
            )

ams = [AssManager(config) for config in configs]
for am in ams:
    am.run()