from flask import Blueprint, request
from deepface.api.src.modules.core import service
from deepface.commons.logger import Logger

logger = Logger(module="api/src/routes.py")

blueprint = Blueprint("routes", __name__)


@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"


@blueprint.route("/fork/represent", methods=["POST"])
def fork_represent():
    import os
    import json
    from flask import jsonify
    from deepface.commons import folder_utils

    root_dir = folder_utils.get_deepface_home() + "/.deepface"

    if "image_data" not in request.files:
        return jsonify({"error": "No image_data part"}), 400
    if "param_json" not in request.form:
        return jsonify({"error": "No param_json in form"}), 400

    file = request.files['image_data']
    params = json.loads(request.form.get("param_json"))

    if file.filename == '':
        return jsonify({"error": "No filename"}), 400

    if "temporary_path" not in params:
        params["temporary_path"] = root_dir + "/tmp"

    img_path = os.path.join(params["temporary_path"], file.filename)
    file.save(img_path)

    model_name = params["model_name"] if "model_name" in params else "VGG-Face"
    detector_backend = params["detector_backend"] if "detector_backend" in params else "opencv"
    enforce_detection = params["enforce_detection"] if "enforce_detection" in params else True
    align = params["align"] if "align" in params else True

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
