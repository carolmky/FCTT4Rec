from FCTT.main import main
ha={'templates': [],
'mode': 'train',
'resume_training': 'false',
'pilot':' false',
'pretrained_weights': 'null',
'num_users': 'null',
'num_items': 'null',
'num_ratings': 'null',
'num_days': 'null',
'local_data_folder':' ./Data',
'dataset_code': 'ml-1m',
'min_rating': 0,
'min_uc': 5,
'min_sc': 5,
'split':' leave_one_out',
'dataloader_code': 'bert',
'dataloader_random_seed': 0.0,
'train_batch_size': 256,
'val_batch_size': 256,
'test_batch_size': 256,
'train_window': 100,
'dataloader_output_timestamp': 'ture',
'dataloader_output_days': 'ture',
'dataloader_output_user': 'false',
'train_negative_sampler_code': 'random',
'train_negative_sample_size': 0,
'train_negative_sampling_seed': 0,
'test_negative_sampler_code': 'popular',
'test_negative_sample_size': 100,
'test_negative_sampling_seed': 98765,
'trainer_code':' bert',
#device: cuda
'device':' cpu',
'use_parallel': 'true',
'num_workers': 0,
'optimizer': 'Adam',
'lr': 0.001,
'weight_decay': 0.0,
'momentum': 'null',
'decay_step': 25,
'gamma': 1.0,
'clip_grad_norm': 5.0,
'num_epochs': -1,
'log_period_as_iter': 12800,
'metric_ks':
- 1
- 5
- 10
- 20
- 50,
'best_metric': 'NDCG@10',
'saturation_wait_epochs': 20,
'model_code': 'meantime',
'model_init_seed': 0,
'model_init_range': 0.02,
'max_len': 200,
'hidden_units': 64,
'num_blocks': 2,
'num_heads': 2,
'dropout': 0.2,
'mask_prob': 0.2,
'output_info': 'false',
'residual_ln_type': 'pre',
'headtype': 'dot',
'head_use_ln': 'true',
'time_unit_divide': 1,
'freq': 10000,
'tisas_max_time_intervals': 'null',
'marank_max_len': 'null',
'marank_num_att_layers': 'null',
'marank_num_linear_layers': 'null',
'absolute_kernel_types': 'p-d',
'relative_kernel_types': 's-l',
'experiment_root': 'experiments',
'experiment_group': 'test',
'experiment_name': 'meantime',
'wandb_project_name': 'null',
'wandb_run_name': 'null',
'wandb_run_id': 'null',
'meta': 'training'
}

if __name__ == '__main__':
    main(ha)