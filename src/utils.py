import itertools
import cv2
import ultralytics
import yaml
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_classes_metrics(results):
    try:
        class_metrics = pd.DataFrame(
            [results.box.class_result(x) for x in results.names.keys()],
            index=results.names.values(),
            columns=["precision", "recall", "ap50", "ap"],
        )
    except IndexError:
        class_metrics = pd.DataFrame(
            [results.box.class_result(x) for x in results.ap_class_index],
            index=results.ap_class_index,
            columns=["precision", "recall", "ap50", "ap"],
        )

    return class_metrics


def get_global_metrics(results):
    global_metrics = pd.DataFrame(
        [results.box.mean_results()],
        index=["all"],
        columns=["precision", "recall", "ap50", "ap"],
    )

    return global_metrics


def save_custom_metrics(folder_name: str, file_name: str, results):
    if not file_name.endswith(".csv"):
        file_name += ".csv"

    folder_path = Path("runs", "detect", folder_name)
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path.joinpath(file_name)

    class_metrics = get_classes_metrics(results=results)
    global_metrics = get_global_metrics(results=results)

    metrics_df = pd.concat([class_metrics, global_metrics], axis=0)
    metrics_df.index.name = "class"
    metrics_df.to_csv(file_path)
    logger.info(f"Custom metric file saved as {file_path}")


def preprocess_image(image_path, target_size=(640, 640)):
    try:
        image = cv2.imread(image_path)
        processed_image = cv2.resize(image, target_size)

        return processed_image

    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None


def preprocess_images_in_folder(input_folder, output_folder, target_size=(640, 640)):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preprocessing files in folder {input_folder}...")
    for filepath in Path(input_folder).iterdir():
        processed_image = preprocess_image(filepath, target_size)

        try:
            cv2.imwrite(output_folder.joinpath(filepath.name), processed_image)
        except Exception as e:
            logger.error(
                f"Error saving image {output_folder.joinpath(filepath.name)}: {e}"
            )
    logger.info(f"Processed images saved in folder {output_folder}")


def grid_search_train(
    model: ultralytics.models.yolo.model.YOLO,
    data: str,
    param_grid: dict = {},
    model_name: str = "model_combination",
):
    param_combinations = list(itertools.product(*param_grid.values()))

    metric_df = pd.DataFrame()
    params_list = []

    logger.info(f"{len(param_combinations)} hyperparameters combinations")
    for i, combination in enumerate(param_combinations, start=0):
        params = dict(zip(param_grid.keys(), combination))
        logger.info(f"Fitting combination {i+1} of {len(param_combinations)}: {params}")
        params_list.append(params)

        results = model.train(data=data, name=f"{model_name}_{i}", **params)
        save_custom_metrics(
            folder_name=f"{model_name}_{i}",
            file_name="custom_validation_metrics",
            results=results,
        )
        metric_df = pd.concat(
            [metric_df, get_global_metrics(results)], ignore_index=True
        )

    metric_df = pd.concat([metric_df, pd.Series(params_list, name="params")], axis=1)

    return metric_df
