from trackmania_structured_vs_code import (
    load_and_prepare_data,
    detect_plateau,
    plot_improvement_curve,
    train_linear_model,
    evaluate_model
)
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRACK_FOLDER = r"C:\Users\mark\OneDrive\Skrivebord\Synopsis\archive\ABC\A-0"

def main():
    all_files = [os.path.join(TRACK_FOLDER, f"{i}.csv") for i in range(1, 11)]
    replay_data = []

    for file in all_files:
        data = load_and_prepare_data(file)
        data["replay_id"] = int(os.path.basename(file).replace(".csv", ""))
        replay_data.append(data)

    summary = pd.DataFrame(replay_data).sort_values("replay_id")
    print(summary)

    # Plot forbedringskurve
    plot_improvement_curve(
        replay_ids=summary["replay_id"],
        lap_times=summary["lap_time"],
        track_id="1"
    )

    # Detekter plateau
    plateau = detect_plateau(summary["lap_time"])
    print("Plateau (True = under 1% forbedring):")
    print(plateau)

    # Regression
    X = summary[["mean_speed", "gas_time", "brake_time", "slide_events"]]
    y = summary["lap_time"]

    model = train_linear_model(X, y)
    print("Lineær model trænet.")

    # Evaluering af overfitting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    evaluate_model(X_scaled, y)

if __name__ == "__main__":
    main()



