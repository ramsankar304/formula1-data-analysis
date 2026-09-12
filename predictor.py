import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
race_id = 1014
driver_ids = [1, 822, 830, 20, 844, 842, 825, 832, 826, 154]
driver_names = {
    1: 'HAM', 822: 'BOT', 830: 'VER', 20: 'VET', 844: 'LEC',
    842: 'GAS', 825: 'MAG', 832: 'SAI', 826: 'KVY', 154: 'GRO'
}
lap_times = pd.read_csv('lap_times.csv')
pit_stops = pd.read_csv('pit_stops.csv')
results = pd.read_csv('results.csv')

# Keep only our chosen race and the 10 target drivers
laps = lap_times[(lap_times['raceId'] == race_id) & (lap_times['driverId'].isin(driver_ids))].copy()
pits = pit_stops[(pit_stops['raceId'] == race_id) & (pit_stops['driverId'].isin(driver_ids))].copy()
grid_info = results[(results['raceId'] == race_id) & (results['driverId'].isin(driver_ids))][['driverId', 'grid']]

laps['code'] = laps['driverId'].map(driver_names)
total_starting_laps = len(laps)

pit_laps = pits[['driverId', 'lap']].copy()
pit_laps['is_pit_lap'] = 1
laps = laps.merge(pit_laps, on=['driverId', 'lap'], how='left')
laps['is_pit_lap'] = laps['is_pit_lap'].fillna(0)


out_laps = pits[['driverId', 'lap']].copy()
out_laps['lap'] = out_laps['lap'] + 1
out_laps['is_out_lap'] = 1
laps = laps.merge(out_laps, on=['driverId', 'lap'], how='left')
laps['is_out_lap'] = laps['is_out_lap'].fillna(0)


laps['is_lap_1'] = laps['lap'] == 1
laps['is_safety_car'] = (laps['lap'] >= 44) & (laps['lap'] <= 52)

bad_laps = (laps['is_pit_lap'] == 1) | (laps['is_out_lap'] == 1) | (laps['is_lap_1'] == True) | (laps['is_safety_car'] == True)
clean_laps = laps[~bad_laps].copy()

print("--- Data Cleaning Summary ---")
print(f"Total starting laps: {total_starting_laps}")
print(f"Removed laps: {bad_laps.sum()}")
print(f"Clean laps remaining: {len(clean_laps)}\n")

clean_laps = clean_laps.sort_values(by=['driverId', 'lap'])
stint_list = []
tire_age_list = []


for driver in driver_ids:
    driver_laps = clean_laps[clean_laps['driverId'] == driver]
    driver_pits = pits[pits['driverId'] == driver]['lap'].values
    
    current_stint = 1
    current_age = 0
    
    
    for index, row in driver_laps.iterrows():
        lap_no = row['lap']
        stops_done = 0
        for pit_lap in driver_pits:
            if lap_no > pit_lap:
                stops_done += 1
                
        new_stint = stops_done + 1
        
        if new_stint != current_stint:
            current_stint = new_stint
            current_age = 1
        else:
            current_age += 1
            
        stint_list.append(current_stint)
        tire_age_list.append(current_age)

clean_laps['stint'] = stint_list
clean_laps['tire_age'] = tire_age_list
clean_laps = clean_laps.merge(grid_info, on='driverId', how='left')
clean_laps['lap_time_sec'] = clean_laps['milliseconds'] / 1000.0
train_data = clean_laps[clean_laps['stint'] < 3].copy()
test_data = clean_laps[clean_laps['stint'] == 3].copy()

y_train = train_data['lap_time_sec']
y_test = test_data['lap_time_sec']

lr_base = LinearRegression().fit(train_data[['grid', 'lap']], y_train)
rf_base = RandomForestRegressor(n_estimators=100, random_state=42).fit(train_data[['grid', 'lap']], y_train)
lr_tire = LinearRegression().fit(train_data[['grid', 'tire_age']], y_train)
rf_tire = RandomForestRegressor(n_estimators=100, random_state=42).fit(train_data[['grid', 'tire_age']], y_train)
combo_model = LinearRegression().fit(train_data[['grid', 'lap', 'tire_age']], y_train)

def print_evaluation(model_name, predictions):
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    print(f"{model_name:<35} | RMSE: {rmse:.3f}s | MAE: {mae:.3f}s")

print("--- Model Performance on Final Test Stint ---")
print_evaluation("Baseline LR (grid + lap)", lr_base.predict(test_data[['grid', 'lap']]))
print_evaluation("Baseline RF (grid + lap)", rf_base.predict(test_data[['grid', 'lap']]))
print_evaluation("Tire Age LR (grid + tire_age)", lr_tire.predict(test_data[['grid', 'tire_age']]))
print_evaluation("Tire Age RF (grid + tire_age)", rf_tire.predict(test_data[['grid', 'tire_age']]))
print_evaluation("Combined LR (grid + lap + tire_age)", combo_model.predict(test_data[['grid', 'lap', 'tire_age']]))

print("\nCombined Model Coefficients:")
print(f"Grid Position: {combo_model.coef_[0]:.4f}")
print(f"Lap Number (Fuel Burn Effect): {combo_model.coef_[1]:.4f}")
print(f"Tire Age (Degradation Effect): {combo_model.coef_[2]:.4f}\n")

ver_data = test_data[test_data['code'] == 'VER'].sort_values('lap').copy()
ver_data['pred'] = combo_model.predict(ver_data[['grid', 'lap', 'tire_age']])

plt.figure(figsize=(9, 5))
plt.plot(ver_data['tire_age'], ver_data['lap_time_sec'], 'o-', label='Actual Lap Times', color='#1f77b4')
plt.plot(ver_data['tire_age'], ver_data['pred'], 's--', label='Predicted Lap Times (Combined Model)', color='#d62728')
plt.title("Max Verstappen: Actual vs Predicted Lap Times (Final Stint)")
plt.xlabel("Tire Age (Laps Completed on Compound)")
plt.ylabel("Lap Time (Seconds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('verstappen_stint_degradation.png')
plt.show()