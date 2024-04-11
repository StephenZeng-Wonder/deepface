from deepface import DeepFace

# pylint: disable=broad-except

def find(img_path, db_path, model_name, detector_backend, enforce_detection, align):
    try:
        result = {}
        similarities = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
        rst = []
        for s in similarities:
            e = {"identity": s.identity, "distance": s.distance, "threshold": s.threshold, "target_x": s.target_x,
                 "target_y": s.target_y, "target_w": s.target_w, "target_h": s.target_h}
            rst.append(e)
        result["results"] = rst
        return result
    except Exception as err:
        return {"error": f"Exception while finding: {str(err)}"}, 400


def represent(img_path, model_name, detector_backend, enforce_detection, align):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        return {"error": f"Exception while representing: {str(err)}"}, 400


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
        )
        return obj
    except Exception as err:
        return {"error": f"Exception while verifying: {str(err)}"}, 400


def analyze(img_path, actions, detector_backend, enforce_detection, align):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
        )
        result["results"] = demographies
        return result
    except Exception as err:
        return {"error": f"Exception while analyzing: {str(err)}"}, 400
