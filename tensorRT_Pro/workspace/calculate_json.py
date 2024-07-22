from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Run COCO mAP evaluation
# Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

annotations_path = "/datasets/datasets/new_dataset/rename/annotations/calib_test_1.json"
results_file = "yolov5_trimmed_qat_noqdq.prediction.json"
cocoGt = COCO(annotation_file=annotations_path)
cocoDt = cocoGt.loadRes(results_file)
imgIds = sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
