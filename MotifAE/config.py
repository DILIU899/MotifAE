my_config = {
    "stage":'representative', 

    "layer": 33,
    "plm_name": 'esm2_t33_650M_UR50D',
    "embed_logit_path_representative": '',  # TODO, path to your precomputed embeddings

    "df_path_representative": '',   # TODO, path to your training metadata file
    "df_name_col_representative": 'uniprot', 

    "save_dir": '', # TODO, path to your model storage
    
    "device": "cuda:0",
    "batch_size": 40,
    "batch_token": None,
    "noise_sd": None,
    "dataloader_num_workers": 10,
    "activation_dim": 1280,
    "dict_size": int(1280*32),
    "resample_steps": None, # resample the weight for the feature every resample_steps
    "resample_training_steps": 0, # only resample the weight in the first resample_steps_training_percent*total_steps
    "tied": False,
    
    "lr": 1e-3,
    "warmup_steps": 500, # learning rate warmup

    "l1_penalty": 0.04,
    "smooth_penalty": 1,
    "l1_annealing_steps": 5000,
    "seed": 42,

    "n_epoch_representative": 2,
    "n_epoch_human": 10,
    "start_step_representative": 0,
    "start_step_human": 110_000,
    "save_steps_representative": 5_000,
    "save_steps_human": 5_000,
    "log_steps": 20,
}
