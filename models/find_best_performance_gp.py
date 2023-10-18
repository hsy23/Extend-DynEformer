# Importing necessary libraries
import pandas as pd
import numpy as np

# Data for the table
data = {
    'gp': [50, 150, 250, 350, 450, 550, 650, 750, 750, 800, 900, 900, 950, 1000, 1000, 1000, 1500, 1500, 2000],
    'MSE': [0.0732, 0.0695, 0.0705, 0.0707, 0.0683, 0.0696, 0.0696, 0.0724, 0.0723, 0.0739, 0.0705, 0.0747, 0.0716, 0.0682, 0.0706, 0.0709, 0.0761, 0.0767, 0.0711],
    'MAE': [0.1417, 0.1349, 0.1383, 0.1396, 0.1372, 0.1400, 0.1317, 0.1366, 0.1421, 0.1377, 0.1391, 0.1425, 0.1375, 0.1331, 0.1416, 0.1424, 0.1468, 0.1435, 0.1375],
    'MAPE': [27.73, 29.46, 27.91, 27.28, 27.87, 27.29, 25.66, 28.10, 28.08, 27.13, 27.84, 27.67, 26.81, 26.47, 28.29, 27.93, 28.47, 26.84, 26.28]
}

# Creating the DataFrame
df = pd.DataFrame(data)

# Grouping by 'gp' and calculating the mean for each group
df_grouped = df.groupby('gp').mean().reset_index()

# Finding the winners for each metric
winners = {
    'MSE': df_grouped.loc[df_grouped['MSE'].idxmin()]['gp'],
    'MAE': df_grouped.loc[df_grouped['MAE'].idxmin()]['gp'],
    'MAPE': df_grouped.loc[df_grouped['MAPE'].idxmin()]['gp']
}

# Counting the winners
winner_count = {gp: 0 for gp in df_grouped['gp']}
for metric, gp in winners.items():
    winner_count[gp] += 1

# Finding the overall winner
overall_winner = max(winner_count, key=winner_count.get)

df_grouped, winners, winner_count, overall_winner
