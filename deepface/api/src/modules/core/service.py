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
            if s.index.stop == 0: continue
            e = {"identity": str(s.identity[0]),
                 "distance": str(s.distance[0]),
                 "target_x": str(s.target_x[0]),
                 "target_y": str(s.target_y[0]),
                 "target_w": str(s.target_w[0]),
                 "target_h": str(s.target_h[0]),
                 "threshold": str(s.threshold[0])}
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
