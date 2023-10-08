from assmanager import AssManager

inflation_values = [1.00, 1.01]
inflation_sequences = ['before_DA']
ensemble_size = 40
configs = []

for inf in inflation_values:
    for seq in inflation_sequences:
        configs.append(
            {
                'model_params': {
                    'forcing': 15.0,
                    'time_steps': 200 * 360 * 5,
                },
                
                'DA_params': {
                    'time_steps': 200 * 360 * 5,
                },
                
                'DA_config': {
                    'ensemble_size': ensemble_size,
                    'inflation_factor': inf,
                    'inflation_sequence': seq,
                },
                
                'DA_option': {
                    'save_kalman_gain': True,
                    'save_prior_ensemble': True,
                    'save_analysis_ensemble': True,
                    'file_save_option': 'single_file',
                },

                'Experiment_option': {
                    'experiment_name': f'F15_inf_{inf}_{seq}_sz{ensemble_size}_5y',
                    'result_save_path': '/mnt/pve_nfs/zyzhao/L05_experiments',
                }
            }
        )

ams = [AssManager(config) for config in configs]
for am in ams:
    am.run()
    