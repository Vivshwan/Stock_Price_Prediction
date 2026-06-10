from tsla_preprocessing import (
    load_tsla_data,
    handle_missing_values,
    select_target_feature,
    scale_series,
    create_time_series_sequences,
    split_train_test,
    evaluate_and_plot,
)
import os


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "TSLA.csv")
    df = load_tsla_data(dataset_path)
    df = handle_missing_values(df)
    target_df = select_target_feature(df, target_column="Adj Close")

    scaled_values, scaler = scale_series(target_df)
    window_size = 60
    forecast_horizon = 1

    X, y = create_time_series_sequences(
        scaled_values, window_size=window_size, forecast_horizon=forecast_horizon
    )

    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio=0.8)

    # Evaluate saved models
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".h5") or model_file.endswith(".keras"):
            model_path = os.path.join(models_dir, model_file)
            model_name = os.path.splitext(model_file)[0]
            evaluate_and_plot(model_path, X_test, y_test, scaler, model_name)


if __name__ == "__main__":
    main()
