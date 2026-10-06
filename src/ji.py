import json
import os
import cv2
import numpy as np
from typing import Any, Dict, Optional
from ultralytics import YOLO
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

_CV2_DRAW_FUNCTIONS = [
    'line', 'arrowedLine', 'rectangle', 'circle', 'ellipse', 'ellipse2Poly',
    'fillConvexPoly', 'fillPoly', 'polylines', 'drawContours', 'drawMarker',
    'putText',
]

_original = {}
for _name in _CV2_DRAW_FUNCTIONS:
    _fn = getattr(cv2, _name, None)
    if _fn is not None:
        _original[_name] = _fn

        def _make_wrapper(orig_fn):
            def _wrapper(img, *args, **kwargs):
                if isinstance(img, np.ndarray) and not img.flags.writeable:
                    img = img.copy()
                return orig_fn(img, *args, **kwargs)
            return _wrapper

        setattr(cv2, _name, _make_wrapper(_fn))


#获取模型路径
def _resolve_model_path() -> str:
	models_root = "/project/train/models"
	model_dir = os.listdir(models_root)[0]

	if os.path.exists(os.path.join(models_root, model_dir, "weights", "best.engine")):
		return os.path.join(models_root, model_dir, "weights", "best.engine")

	if os.path.exists(os.path.join(models_root, model_dir, "weights", "best_fp32.engine")):
		return os.path.join(models_root, model_dir, "weights", "best_fp32.engine")
	elif os.path.exists(os.path.join(models_root, model_dir, "weights", "best_fp16.engine")):
		return os.path.join(models_root, model_dir, "weights", "best_fp16.engine")
	else:
		return os.path.join(models_root, model_dir, "weights", "best.pt")
	
def init() -> Dict[str, Any]:
	model_path = _resolve_model_path()
	model = YOLO(model_path)
	return {"model": model}

def process_image(handle: Optional[Dict[str, Any]] = None,input_image=None,args=None,**kwargs,):
	if handle is None:
		handle = init()
	model = handle["model"]
	results = model.predict(source=input_image, verbose=False,device="cuda:0",workers=4)

	objects = []

	result = results[0]

	if result.boxes is not None and len(result.boxes) > 0:
		for det in result.boxes.data:
			x1, y1, x2, y2,conf,cls=det.tolist()
			objects.append({
				"x": int(x1),
				"y": int(y1),
				"width": int(x2 - x1),
				"height": int(y2 - y1),
				"confidence": float(conf),
				"name": str(result.names.get(int(cls), int(cls))),
			})


	fake_result = {}
	# fake_result["algorithm_data"] = {
	# 	"is_alert": False,
	# 	"target_count": 0,
	# 	"target_info": [],
	# }

	fake_result["model_data"] = {"objects": objects}

	return json.dumps(fake_result, indent=4)
