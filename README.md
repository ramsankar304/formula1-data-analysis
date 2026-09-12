# Formula 1 Lap Time Predictor

A machine learning project to predict Formula 1 lap times within a single Grand Prix using tire age and stint tracking to study degradation behavior.

---

## 1. Race & Driver Selection
- **Grand Prix:** 2019 Spanish Grand Prix (Circuit de Barcelona-Catalunya, Race ID: `1014`)
- **Track Conditions:** Completely dry with no red flags throughout the race.
- **Drivers Analyzed:** The top 10 lead-lap finishers: `HAM`, `BOT`, `VER`, `VET`, `LEC`, `GAS`, `MAG`, `SAI`, `KVY`, and `GRO`.

---

## 2. Data Cleaning & Preprocessing
To isolate tire degradation from unrelated external factors, non-representative laps were filtered out:
- **Pit-in Laps:** Laps where cars entered the pit lane (slowed down by the pit limiter).
- **Pit-out Laps:** The lap immediately following a pit stop (compromised by cold tires).
- **Lap 1:** Standing start and first-lap traffic bunching.
- **Safety Car Window (Laps 44–52):** Slow neutralized pacing following the Norris/Stroll collision.

**Cleaning Summary:**
- Total Initial Laps: `660`
- Removed Laps: `123` (18.6%)
- Clean Laps Retained: `537`

---

## 3. Feature Engineering & Validation Setup
- **Tire Age Calculation:** Engineered using sequential loops that count laps completed on a specific tire compound and reset to 1 immediately following each pit stop.
- **Stint-Based Split:** Stints 1 and 2 are used for model training, while Stint 3 (the final stint) is reserved for testing. This avoids temporal data leakage caused by random train/test splitting across adjacent laps.

---

## 4. Model Evaluation & Benchmark
Models were trained and evaluated on the unseen final stint using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE):

| Model | Feature Set | RMSE (sec) | MAE (sec) |
| :--- | :--- | :---: | :---: |
| **Baseline Linear Regression** | `grid` + `lap` | 1.056s | 0.924s |
| **Baseline Random Forest** | `grid` + `lap` | 1.721s | 1.598s |
| **Tire Age Linear Regression** | `grid` + `tire_age` | 2.255s | 2.162s |
| **Tire Age Random Forest** | `grid` + `tire_age` | 2.193s | 2.105s |
| **Combined Linear Regression** | `grid` + `lap` + `tire_age` | **1.030s** | **0.887s** |

### Key Analytical Insight
Raw `tire_age` alone loses to the baseline because it resets after every stop, losing track of continuous fuel burn-off. When combined (`grid + lap + tire_age`), the effects separate cleanly:
- **Lap Number Coefficient:** ~`-0.043s` (the car becomes lighter and faster as fuel burns).
- **Tire Age Coefficient:** ~`+0.024s` (the car loses pace as the tire compound wears down).

---

## 5. Visualization
The repository includes visual verification plotting predicted vs. actual lap times across Max Verstappen's (`VER`) final stint, demonstrating that the model tracks the overall pace degradation trend.
