import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore

class TrafficModel:
    def __init__(self, time_steps=10):
        self.time_steps = time_steps
        self.model = self._build_model()
        
    def _build_model(self):
        """Construye el modelo LSTM."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.time_steps, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X, y, epochs=50, batch_size=32):
        """Entrena el modelo."""
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def predict(self, X):
        """Realiza predicciones."""
        return self.model.predict(X)
