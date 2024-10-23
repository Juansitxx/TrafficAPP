import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generar fechas y horas
start_date = datetime(2024, 1, 1, 6, 0)  # Comenzar a las 6 AM
dates = [start_date + timedelta(minutes=15*i) for i in range(96)]  # Datos cada 15 minutos por 24 horas

# Generar datos de densidad de tráfico
def generate_traffic_density(time):
    hour = time.hour
    # Simular horas pico (morning rush: 7-9 AM, evening rush: 5-7 PM)
    base = 20  # Densidad base
    if 7 <= hour <= 9:
        base = 60  # Alta densidad en hora pico de la mañana
    elif 17 <= hour <= 19:
        base = 55  # Alta densidad en hora pico de la tarde
    elif 11 <= hour <= 15:
        base = 35  # Densidad media durante el día
    
    # Añadir algo de variación aleatoria
    noise = np.random.normal(0, 5)
    density = max(0, base + noise)  # Asegurar que no sea negativo
    return round(density, 2)

# Crear el DataFrame
data = {
    'timestamp': dates,
    'density': [generate_traffic_density(date) for date in dates]
}

df = pd.DataFrame(data)

# Guardar a CSV
df.to_csv('data/sample_traffic_data.csv', index=False)

# Mostrar las primeras filas
print("Primeras filas del dataset generado:")
print(df.head())