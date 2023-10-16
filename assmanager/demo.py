from assmanager import AssManager

inflation_values = [1.0, 1.05]
inflation_sequences = ['before_DA']
ensemble_size = 2000
forcings = [15]
time_steps = 200 * 360 * 5
# time_steps = 200 * 20
configs = []
localization_method = None if ensemble_size >= 2000 else 'GC'
file_save_option = 'multiple_files' if ensemble_size >= 2000 else 'single_file'

for inf in inflation_values:
    for seq in inflation_sequences:
        for forcing in forcings:
            configs.append(
                {
                    'model_params': {
                        'forcing': forcing,
                        # 'time_steps': time_steps,
                    },
                    
                    'DA_params': {
                        'time_steps': time_steps,
                    },
                    
                    'DA_config': {
                        'ensemble_size': ensemble_size,
                        'inflation_factor': inf,
                        'inflation_sequence': seq,
                        'localization_method': localization_method,
                        'filter': 'EnKF',
                    },
                    
                    'DA_option': {
                        'save_kalman_gain': True,
                        'save_prior_ensemble': True,
                        'save_analysis_ensemble': True,
                        'file_save_option': file_save_option,
                        # 'file_save_option': 'single_file',
                    },

                    'Experiment_option': {
                        'experiment_name': f'EnKF_F{forcing}_inf_{inf}_loc_{localization_method}_{seq}_sz{ensemble_size}_5y',
                        'result_save_path': '/mnt/ssd/L05_experiments',
                    }
                }
            )

ams = [AssManager(config) for config in configs]
for am in ams:
    am.run()