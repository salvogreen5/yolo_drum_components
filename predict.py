from ultralytics import YOLO
from pathlib import Path
import logging
from src.utils import read_config, preprocess_images_in_folder

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Job START")

    # config file
    config = read_config(Path("config", "predict_config.yaml"))

    data_folder = Path(config["DATA_FOLDER"])
    input_folder = data_folder.joinpath(config["INPUT_IMAGES_FOLDER"])
    processed_folder = data_folder.joinpath(config["PROCESSED_IMAGES_FOLDER"])
    output_folder_name = config["PREDICTED_IMAGES_FOLDER"]
    trained_model_path = Path("runs", "detect", config["MODEL_NAME"])
    trained_model_weights = trained_model_path.joinpath("weights", "best.pt")
    imgsize = config["IMGSZ"]

    # load trained model
    drums_model = YOLO(trained_model_weights)
    logger.info(f"Loaded YOLO model: {trained_model_weights}")
    logger.info(
        f"YOLO model detected classes: {sorted(list(drums_model.names.values()))}"
    )

    # load and preprocess images
    preprocess_images_in_folder(
        input_folder=input_folder,
        output_folder=processed_folder,
        target_size=(imgsize, imgsize),
    )

    # make predictions
    logger.info(f"Predict images in {processed_folder}...")
    predict = drums_model(
        source=processed_folder,
        save=True,
        imgsz=imgsize,
        project=data_folder,
        name=output_folder_name,
    )
    logger.info(f"Predict... DONE")


if __name__ == "__main__":
    main()
