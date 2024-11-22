# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
MLflow Logging for Ultralytics YOLO.

This module enables MLflow logging for Ultralytics YOLO. It logs metrics, parameters, and model artifacts.
For setting up, a tracking URI should be specified. The logging can be customized using environment variables.

Commands:
    1. To set a project name:
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument

    2. To set a run name:
        `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument

    3. To start a local MLflow server:
        mlflow server --backend-store-uri runs/mlflow
       It will by default start a local server at http://127.0.0.1:5000.
       To specify a different URI, set the MLFLOW_TRACKING_URI environment variable.

    4. To kill all running MLflow server instances:
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr
from ultralytics import YOLO
import torch
import struct
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

import psycopg2


class Database:
    def __init__(self):
        self.db = psycopg2.connect(
            host="192.168.10.42",
            port="42010",
            dbname="mlops",
            user="nextk",
            password="nextknextk",
        )
        self.cursor = self.db.cursor()
        return

    def __del__(self):
        self.db.close()
        self.cursor.close()
        return

    def execute(self, query):
        self.cursor.execute(query)
        return

    def fetchall(self):
        return self.cursor.fetchall()

    def commit(self):
        self.db.commit()
        return


try:
    import os

    assert not TESTS_RUNNING or "test_mlflow" in os.environ.get("PYTEST_CURRENT_TEST", "")  # do not log pytest
    assert SETTINGS["mlflow"] is True  # verify integration is enabled
    import mlflow

    assert hasattr(mlflow, "__version__")  # verify package is not directory
    from pathlib import Path

    PREFIX = colorstr("MLflow: ")

except (ImportError, AssertionError):
    mlflow = None


def sanitize_dict(x):
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


def on_pretrain_routine_end(trainer):
    """
    Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters
    from the trainer.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.

    Global:
        mlflow: The imported mlflow module to use for logging.

    Environment Variables:
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
        MLFLOW_KEEP_RUN_ACTIVE: Boolean indicating whether to keep the MLflow run active after the end of training.
    """
    global mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/YOLOv8"
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))

        mlflow.log_artifact(trainer.data["yaml_file"], "data")
        if trainer.data.get("train") is not None:
            train_path = trainer.data.get("train")
            if os.path.isfile(train_path) and train_path.endswith(".txt"):
                mlflow.log_artifact(train_path, "data")
        if trainer.data.get("val") is not None:
            val_path = trainer.data.get("val")
            if os.path.isfile(val_path) and val_path.endswith(".txt"):
                mlflow.log_artifact(val_path, "data")
        if trainer.data.get("test") is not None:
            test_path = trainer.data.get("test")
            if os.path.isfile(test_path) and test_path.endswith(".txt"):
                mlflow.log_artifact(test_path, "data")
        mlflow.pytorch.log_model(torch.nn.Module(), "model")

        db = Database()
        try:
            db.execute(f"UPDATE training SET model_id='{active_run.info.run_id}' WHERE id='{run_name}'")
            db.commit()
        except Exception as e:
            print("[DB Update Error] ", e)

    except Exception as e:
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n" f"{PREFIX}WARNING ⚠️ Not tracking this run")


def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(
            metrics={
                **sanitize_dict(trainer.lr),
                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
            step=trainer.epoch,
        )

        run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
        db = Database()
        try:
            db.execute(f"UPDATE training SET current_epoch={trainer.epoch+1} WHERE id='{run_name}'")
            db.commit()
        except Exception as e:
            print("[DB Update Error] ", e)


def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if not mlflow:
        return

    # region convert (pt) model into wts format
    pt_model_path = str(trainer.best)
    wts_model_path = os.path.splitext(pt_model_path)[0] + ".wts"
    model = torch.load(pt_model_path, map_location="cpu")

    if model["model"] is None:
        model["model"] = YOLO(pt_model_path).model

    model = model["model"].float()
    delattr(model.model[-1], "anchors")

    model.to("cpu").eval()
    with open(wts_model_path, "w") as f:
        f.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {} ".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
    # endregion

    # region encrypt wts model
    _key = bytes([0xFF, 0x01, 0xEE, 0xD7, 0xC4, 0xB5, 0x02, 0x07, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0xAE, 0xFF])
    _iv = bytes([0x1F, 0x5A, 0x9F, 0x0B, 0x3F, 0xFC, 0xFF, 0xCD, 0xFF, 0x00, 0x45, 0x78, 0x26, 0x74, 0x69, 0xFF])

    enc_file_path = os.path.splitext(wts_model_path)[0] + "_enc.wts"
    with open(wts_model_path, "r", newline="\r\n") as f:
        buffer = f.read().encode("utf-8")

    aes = AES.new(_key, AES.MODE_CBC, iv=_iv)

    block_size = 16
    padded_buffer = pad(buffer, block_size)
    encrypted_text = aes.encrypt(padded_buffer)

    with open(enc_file_path, "wb") as f:
        f.write(encrypted_text)
    # endregion

    mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt
    for f in trainer.save_dir.glob("*"):  # log all other files in save_dir
        if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
            mlflow.log_artifact(str(f))
    keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    if keep_run_active:
        LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")
    else:
        mlflow.end_run()
        LOGGER.debug(f"{PREFIX}mlflow run ended")

    LOGGER.info(
        f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n{PREFIX}disable with 'yolo settings mlflow=False'"
    )


callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if mlflow
    else {}
)
