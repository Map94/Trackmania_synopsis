# ----------------------------------------
# IMPORTS
# ----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# ----------------------------------------
# 1. DATAFORBEREDELSE
# ----------------------------------------

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    # Fjern whitespaces og gør kolonnenavne små
    df.columns = [col.strip().lower() for col in df.columns]

    # Definer forventede kolonner (små bogstaver, ingen mellemrum)
    expected_cols = ['vx', 'vy', 'vz', 'time', 'gas', 'brake']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    # Tilføj manglende kolonner med 0
    for col in missing_cols:
        df[col] = 0

    lap_time = df['time'].max() / 1000.0
    speeds = (df[['vx', 'vy', 'vz']]**2).sum(axis=1)**0.5
    mean_speed = speeds.mean()
    max_speed = speeds.max()
    gas_time = df['gas'].sum() * 0.01
    brake_time = df['brake'].sum() * 0.01
    # Find alle kolonner der matcher 'is_sliding' (også hvis der er mellemrum)
    slide_cols = [col for col in df.columns if 'is_sliding' in col.replace(" ", "")]
    slide_events = df[slide_cols].any(axis=1).sum() if slide_cols else 0

    return {
        "lap_time": lap_time,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "gas_time": gas_time,
        "brake_time": brake_time,
        "slide_events": slide_events
    }

# ----------------------------------------
# 2. PLATEAU-ANALYSE
# ----------------------------------------

def detect_plateau(lap_times):
    rolling = lap_times.rolling(window=3).mean()
    improvement = rolling.pct_change()
    plateau = improvement.abs() < 0.01
    return plateau

# ----------------------------------------
# 3. VISUALISERING
# ----------------------------------------

def plot_improvement_curve(replay_ids, lap_times, track_id):
    plt.plot(replay_ids, lap_times, marker='o')
    plt.title(f'Forbedringskurve - Track {track_id}')
    plt.xlabel('Replay')
    plt.ylabel('Rundetid (sek)')
    plt.grid(True)
    plt.show()

# ----------------------------------------
# 4. MODELTRÆNING
# ----------------------------------------

def train_linear_model(X, y):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    pipeline.fit(X, y)
    return pipeline

def train_regularized_model(X_scaled, y):
    params = {"alpha": [0.1, 1.0, 10.0]}
    model = GridSearchCV(Ridge(), params, cv=5)
    model.fit(X_scaled, y)
    return model

# ----------------------------------------
# 5. OVERFITTING CHECK
# ----------------------------------------

def evaluate_model(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    score = ridge.score(X_test, y_test)
    print("R^2 på testdata:", score)
    return score

# ----------------------------------------
# 6. KØRSEL AF HELE PIPELINE
# ----------------------------------------

def run_pipeline(filepath, track_id):
    df = load_and_prepare_data(filepath)
    lap_times = df['lap_time']
    replay_ids = df['replay_id']
    
    # 2. Detekter plateau
    plateau = detect_plateau(lap_times)
    df['plateau'] = plateau
    
    # 3. Visualiser forbedringskurve
    plot_improvement_curve(replay_ids, lap_times, track_id)
    
    # 4. Træn model
    features = df[['vx', 'vy', 'vz', 'gas', 'brake']]
    target = df['lap_time']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    linear_model = train_linear_model(X_train, y_train)
    X_scaled = StandardScaler().fit_transform(X_train)
    regularized_model = train_regularized_model(X_scaled, y_train)
    
    # 5. Evaluer model
    print("Evaluering af lineær model:")
    evaluate_model(X_train, y_train)
    
    print("Evaluering af regulariseret model:")
    X_scaled_test = StandardScaler().fit_transform(X_test)
    evaluate_model(X_scaled_test, y_test)


