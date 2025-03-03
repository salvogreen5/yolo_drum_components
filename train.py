from ultralytics import YOLO
from pathlib import Path
import logging
from src.utils import read_config, save_custom_metrics, grid_search_train

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Job START")

    # config file
    config = read_config(Path("config", "train_config.yaml"))

    model_folder_path = Path(config["MODEL_FOLDER"])
    yolo_model = config["YOLO_MODEL_TYPE"]
    model_path = model_folder_path.joinpath(yolo_model)
    model_name = config["MODEL_NAME"]
    test_data = config["TEST_DATA_YAML_FILE"]
    grid_search_metric = config.get("GRID_SEARCH_METRIC", "ap")
    baseline_model_folder = config["BASELINE_MODEL_TEST_FOLDER"]

    # load pretrained model
    drums_model = YOLO(model_path)
    baseline_model = YOLO(model_path)
    logger.info(f"Loaded YOLO model: {yolo_model}")
    logger.info(
        f"YOLO model detected classes: {sorted(list(baseline_model.names.values()))}"
    )

    # YOLO baseline model on test data
    logger.info(f"Evaluating YOLO baseline model test data performance... DONE")
    baseline_test_results = baseline_model.val(
        data=test_data, name=baseline_model_folder, verbose=False
    )
    save_custom_metrics(
        folder_name=baseline_model_folder,
        file_name=config["CUSTOM_METRICS_FILENAME"],
        results=baseline_test_results,
    )
    logger.info(f"Evaluating YOLO baseline model test data performance... DONE")

    # grid search train
    logger.info(f"Training drums model with grid search...")
    grid_search_metrics = grid_search_train(
        data=config["TRAIN_DATA_YAML_FILE"],
        model=drums_model,
        model_name=model_name,
        param_grid=config["GRID_SEARCH_PARAMS"],
    )
    logger.info(f"Training drums model with grid search... DONE")

    # get best model
    best_combination = grid_search_metrics[grid_search_metric].idxmax()
    best_model_folder = Path(f"runs/detect/{model_name}_{best_combination}")
    best_model_weights = best_model_folder.joinpath("weights", "best.pt")
    best_params = grid_search_metrics.loc[best_combination, "params"]

    logger.info(f"Using '{grid_search_metric}' metric to find best model")
    logger.info(
        f"Best is model at {best_model_folder} with {grid_search_metric}={grid_search_metrics.loc[best_combination, grid_search_metric]:.2f} and params: {best_params}"
    )

    drums_model = YOLO(model=best_model_weights)

    # drums model on test data
    logger.info(f"Evaluating drums model test performance...")

    drums_test_results = drums_model.val(
        data=test_data, name=f"{model_name}_test_results", verbose=False
    )
    save_custom_metrics(
        folder_name=f"{model_name}_test_results",
        file_name="custom_metrics",
        results=drums_test_results,
    )

    logger.info(f"Evaluating drums model test performance... DONE")

    logger.info(f"Job DONE")


if __name__ == "__main__":
    main()
