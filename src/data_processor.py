import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_data(self, data, time_steps=10):
        """Prepara los datos para el modelo LSTM."""
        X, y = [], []
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        for i in range(len(scaled_data) - time_steps):
            X.append(scaled_data[i:(i + time_steps)])
            y.append(scaled_data[i + time_steps])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data):
        """Revierte la normalización de los datos."""
        return self.scaler.inverse_transform(data)