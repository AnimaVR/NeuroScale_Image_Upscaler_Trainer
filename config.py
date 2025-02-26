
training_config = { 
    'mode': 'scratch',       # Training mode: 'scratch' or 'resume'
    'hidden_dim': 1536,      # Hidden dimension for the model ### increases increase GPU memory requirements a lot.
    'n_layers': 4,           # Number of layers in the model
    'num_heads': 8,          # Number of attention heads
    'dropout': 0.2,          # Dropout rate | Consider, if being used as compression. Do we actually WANT to overfit?
    'batch_size': 8,      
    'micro_batch_size': 256, # Micro batch size, this is the height of the image
    'learning_rate': 5e-5,   # Learning rate
    'weight_decay': 1e-5,    # Weight decay for the optimizer
    'n_epochs': 200,          # Number of training epochs
    'output_dim': 768,           # this is width x 3 for RGB  
    'delta': 1,              
    'w1': 1.0,               
    'w2': 1.0, 
    'w3': 1.0, 
    'use_multi_gpu' : False,   
    'num_gpus' : 1,               
    'warmup_epochs': 0, 
    'input_dim': 768,    # this is width x 3 for RGB  - we stretch the 64x64 image up to create pixillated version of same size as taret
    'frame_size': 256, # this is the height of the image
    'root_dir': r"dataset/data",     
    'model_path': r"out/model.pth",
    'checkpoint_path': r"out/checkpoints/checkpoint.pth", 
}

