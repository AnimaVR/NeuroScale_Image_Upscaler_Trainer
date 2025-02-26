
training_config = { 
    'mode': 'scratch',       # Training mode: 'scratch' or 'resume'
    'hidden_dim': 1024,      # Hidden dimension for the model ### increases increase GPU memory requirements a lot.
    'n_layers': 4,           # Number of layers in the model
    'num_heads': 4,          # Number of attention heads
    'dropout': 0.2,          # Dropout rate
    'batch_size':  64,      # Batch size ## REDUCE THIS IF < 24GB GPU
    'micro_batch_size': 256, # Micro batch size # If you increase this you need to reduce the batch size
    'learning_rate': 1e-4,   # Learning rate
    'weight_decay': 1e-5,    # Weight decay for the optimizer
    'n_epochs': 50,          # Number of training epochs
    'output_dim': 768,             
    'delta': 1,              
    'w1': 1.0,               
    'w2': 1.0, 
    'w3': 1.0, 
    'use_multi_gpu' : False,   
    'num_gpus' : 1,               
    'warmup_epochs': 0, 
    'input_dim': 768,  
    'frame_size': 256,
    'root_dir': r"dataset/data",     
    'model_path': r"out/model.pth",
    'checkpoint_path': r"out/checkpoints/checkpoint.pth", 
}

