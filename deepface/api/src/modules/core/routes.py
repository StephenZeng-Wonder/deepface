from flask import Blueprint, request
from deepface.api.src.modules.core import service
from deepface.commons.logger import Logger

logger = Logger(module="api/src/routes.py")

blueprint = Blueprint("routes", __name__)


@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"


import os
import time
import mediapipe as mp

from deepface.commons import folder_utils
from deepface.commons import keypoint_utils
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

root_dir = folder_utils.get_deepface_home() + "/.deepface"
model_path = root_dir + "/weights/face_landmarker.task"

@blueprint.route("/landmark", methods=["POST"])
def landmark():
    temporary_path = request.args.get("temporary_path", root_dir + "/tmp")
    filename = request.args.get("filename", str(int(time.time() * 1000)) + ".jpg")

    img_path = os.path.join(temporary_path, filename)
    with open(img_path, 'wb') as file:
        file.write(request.data)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1)

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(img_path)
        obj = landmarker.detect(mp_image).face_landmarks[0]
        obj = keypoint_utils.landmark_478_to_keypoint(obj)

    logger.debug(obj)
    os.remove(img_path)

    return obj


@blueprint.route("/find", methods=["POST"])
def find():
    model_name = request.args.get("model_name", "VGG-Face")
    detector_backend = request.args.get("detector_backend", "opencv")
    enforce_detection = request.args.get("enforce_detection", True)
    align = request.args.get("align", True)
    temporary_path = request.args.get("temporary_path", root_dir + "/tmp")
    database_path = request.args.get("database_path", root_dir + "/database")
    filename = request.args.get("filename", str(int(time.time() * 1000)) + ".jpg")

    img_path = os.path.join(temporary_path, filename)
    with open(img_path, 'wb') as file:
        file.write(request.data)

    obj = service.find(
        img_path=img_path,
        db_path=database_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    logger.debug(obj)
    os.remove(img_path)

    return obj



@blueprint.route("/fork/represent", methods=["POST"])
def fork_represent():
    model_name = request.args.get("model_name", "VGG-Face")
    detector_backend = request.args.get("detector_backend", "opencv")
    enforce_detection = request.args.get("enforce_detection", True)
    align = request.args.get("align", True)
    temporary_path = request.args.get("temporary_path", root_dir + "/tmp")
    filename = request.args.get("filename", str(int(time.time() * 1000)) + ".jpg")

    img_path = os.path.join(temporary_path, filename)
    with open(img_path, 'wb') as file:
        file.write(request.data)

    obj = service.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    logger.debug(obj)
    os.remove(img_path)

    return obj


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)

    obj = service.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    distance_metric = input_args.get("distance_metric", "cosine")
    align = input_args.get("align", True)

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)
    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])

    demographies = service.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    logger.debug(demographies)

    return demographies
