echo "Build FP32 Model"

TRTEXEC=/opt/TensorRT-8.6.1.6/bin/trtexec
${TRTEXEC} --onnx=yolov5_trimmed_qat_noqdq.onnx  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=yolov5_trimmed_qat_noqdq.FP32.trtmodel

echo "Build FP16 Model"

TRTEXEC=/opt/TensorRT-8.6.1.6/bin/trtexec
${TRTEXEC} --onnx=yolov5_trimmed_qat_noqdq.onnx  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --fp16 --saveEngine=yolov5_trimmed_qat_noqdq.FP16.trtmodel

echo "Build INT8 Model"

TRTEXEC=/opt/TensorRT-8.6.1.6/bin/trtexec
${TRTEXEC} --onnx=yolov5_trimmed_qat_noqdq.onnx  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --fp16 --int8 --saveEngine=yolov5_trimmed_qat_noqdq.INT8.trtmodel --calib=yolov5_trimmed_qat_precision_config_calib.cache --precisionConstraints=obey --layerPrecisions="/model.24/Reshape":fp16,"/model.24/Transpose":fp16,"/model.24/Sigmoid":fp16,"/model.24/Split":fp16,"/model.24/Mul":fp16,"/model.24/Add":fp16,"/model.24/Pow":fp16,"/model.24/Mul_1":fp16,"/model.24/Mul_3":fp16,"/model.24/Concat":fp16,"/model.24/Concat":fp16,"/model.24/Reshape_1":fp16,"/model.24/Concat_3":fp16,"/model.24/Reshape_2":fp16,"/model.24/Transpose_1":fp16,"/model.24/Sigmoid_1":fp16,"/model.24/Split_1":fp16,"/model.24/Mul_4":fp16,"/model.24/Add_1":fp16,"/model.24/Pow_1":fp16,"/model.24/Mul_5":fp16,"/model.24/Mul_7":fp16,"/model.24/Concat_1":fp16,"/model.24/Concat_1":fp16,"/model.24/Reshape_3":fp16,"/model.24/Concat_3":fp16,"/model.24/Reshape_4":fp16,"/model.24/Transpose_2":fp16,"/model.24/Sigmoid_2":fp16,"/model.24/Split_2":fp16,"/model.24/Mul_8":fp16,"/model.24/Add_2":fp16,"/model.24/Pow_2":fp16,"/model.24/Mul_9":fp16,"/model.24/Mul_11":fp16,"/model.24/Concat_2":fp16,"/model.24/Concat_2":fp16,"/model.24/Reshape_5":fp16,"/model.24/Concat_3":fp16