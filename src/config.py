# src/config.py
import os
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(PROJ_ROOT, 'data/raw/krkopt.data')
output_dir = 'models/'
output_dim = 18
hidden_layers = [(64,32),(64, 64,64,32),(128, 128,128,128,128,128,32)]
learning_rate = 0.01
epochs = 700
lambda_values = [0,0.3,0.7]