# yolov5代码修改
## cuDLA-samples代码拷贝到yolov5中
```
cp -r cuDLA-samples/export/yolov5-qat/* yolov5/
cp -r cuDLA-samples/export/qdq_translator yolov5/
```
+ 添加的scripts为主要的量化调用，quantization为torch和pytorch-quantization的中间工具，qdq_translator为onnx处理代码，common.py就替换了concat为自定义的类
## 库和依赖
+ pytorch-quantization，安装2.1.3版本
```
pip install pytorch-quantization==2.1.3 --extra-index-url https://pypi.ngc.nvidia.com

```
+ 安装cuDLA-samples/export/qdq_translator/requirements.txt
## 数据集准备
+ 数据格式
```
├── images
│   ├── train
│   └── val
├── labels
│   ├── train
│   └── val
├── train2017.txt
└── val2017.txt
```
+ txt文件生成代码
```
import os

save_dir  = "/home/jarvis/Learn/Datasets/VOC_QAT"
train_dir = "/home/jarvis/Learn/Datasets/VOC_QAT/images/train"
train_txt_path = os.path.join(save_dir, "train2017.txt")

with open(train_txt_path, "w") as f:
    for filename in os.listdir(train_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"): # 添加你的图像文件扩展名
            file_path = os.path.join(train_dir, filename)
            f.write(file_path + "\n")

print(f"train2017.txt has been created at {train_txt_path}")

val_dir = "/home/jarvis/Learn/Datasets/VOC_QAT/images/val"
val_txt_path = os.path.join(save_dir, "val2017.txt")

with open(val_txt_path, "w") as f:
    for filename in os.listdir(val_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"): # 添加你的图像文件扩展名
            file_path = os.path.join(val_dir, filename)
            f.write(file_path + "\n")

print(f"val2017.txt has been created at {val_txt_path}")


```
## 数据集调用目标调整
+ scripts/qat.py中121行数据集的调用地址调整
+ models/yolov5s.yaml的类别数调整
# QAT步骤
## 微调
+ 执行微调
```
python scripts/qat.py quantize best.pt --ptq=ptq.pt --qat=qat.pt --cocodir=/datasets/datasets/new_dataset/QAT_dataset2 --eval-ptq --eval-origin
```
+ 修改校准方法：quantization/quantize.py的309行可设置mse、percentile、entropy三种方法
+ 修改数据集加载的workers，在scripts/qat.py的create_dataloader函数
+ 修改scripts/qat.py183行calibrate_model函数的num_batch参数调准校准数据量
## 导出
+ 修改scripts/qat.py153行的导出名称为output
```
# srcipts/qat.py第153行，export_onnx函数
# quantize.export_onnx(model, dummy, file, opset_version=13, 
#     input_names=["images"], output_names=["outputs"], 
#     dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} if dynamic_batch else None
# )
# 修改为：

quantize.export_onnx(model, dummy, file, opset_version=13, 
    input_names=["images"], output_names=["output"], 
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
)

```
+ 修改models/yolo.py中的模型代码
```
# yolov5-7.0/models/yolo.py第60行，forward函数
# bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
# x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
# 修改为：

bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
bs = -1
ny = int(ny)
nx = int(nx)
x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

# yolov5-7.0/models/yolo.py第79行，forward函数
# return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
# 修改为：

return (torch.cat(z, 1),)

```
+ 执行onnx导出,此处可以多设置一个-noanchor
```
python scripts/qat.py export qat.pt --size=640 --save=yolov5_trimmed_qat.onnx --dynamic
```
+ onnx转换,得到noqdq的onnx，校准用的cache，敏感度分析的txt和json
```
python qdq_translator/qdq_translator.py --input_onnx_models=yolov5_trimmed_qat.onnx --output_dir=./ --infer_concat_scales --infer_mul_scales

```
## trt生成
+ 创建build.bash一键生成
```
echo "Build FP32 Model"

TRTEXEC=/opt/TensorRT-8.6.1.6/bin/trtexec
${TRTEXEC} --onnx=yolov5_trimmed_qat_noqdq.onnx  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=yolov5_trimmed_qat_noqdq.FP32.trtmodel

echo "Build FP16 Model"

TRTEXEC=/opt/TensorRT-8.6.1.6/bin/trtexec
${TRTEXEC} --onnx=yolov5_trimmed_qat_noqdq.onnx  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --fp16 --saveEngine=yolov5_trimmed_qat_noqdq.FP16.trtmodel

echo "Build INT8 Model"

TRTEXEC=/opt/TensorRT-8.6.1.6/bin/trtexec
${TRTEXEC} --onnx=yolov5_trimmed_qat_noqdq.onnx  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --fp16 --int8 --saveEngine=yolov5_trimmed_qat_noqdq.INT8.trtmodel --calib=yolov5_trimmed_qat_precision_config_calib.cache --precisionConstraints=obey --layerPrecisions="/model.24/Reshape":fp16,"/model.24/Transpose":fp16,"/model.24/Sigmoid":fp16,"/model.24/Split":fp16,"/model.24/Mul":fp16,"/model.24/Add":fp16,"/model.24/Pow":fp16,"/model.24/Mul_1":fp16,"/model.24/Mul_3":fp16,"/model.24/Concat":fp16,"/model.24/Concat":fp16,"/model.24/Reshape_1":fp16,"/model.24/Concat_3":fp16,"/model.24/Reshape_2":fp16,"/model.24/Transpose_1":fp16,"/model.24/Sigmoid_1":fp16,"/model.24/Split_1":fp16,"/model.24/Mul_4":fp16,"/model.24/Add_1":fp16,"/model.24/Pow_1":fp16,"/model.24/Mul_5":fp16,"/model.24/Mul_7":fp16,"/model.24/Concat_1":fp16,"/model.24/Concat_1":fp16,"/model.24/Reshape_3":fp16,"/model.24/Concat_3":fp16,"/model.24/Reshape_4":fp16,"/model.24/Transpose_2":fp16,"/model.24/Sigmoid_2":fp16,"/model.24/Split_2":fp16,"/model.24/Mul_8":fp16,"/model.24/Add_2":fp16,"/model.24/Pow_2":fp16,"/model.24/Mul_9":fp16,"/model.24/Mul_11":fp16,"/model.24/Concat_2":fp16,"/model.24/Concat_2":fp16,"/model.24/Reshape_5":fp16,"/model.24/Concat_3":fp16
```
+ 参考ptq的方法用tensorRT_Pro库中app_yolo生成
```
test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5_trimmed_qat_noqdq") // 修改1 177行 "yolov7"改成"yolov5_trimmed_qat_noqdq"

static const char *voclabels[] = {"aeroplane",   "bicycle", "bird",   "boat",       "bottle",
                                  "bus",         "car",     "cat",    "chair",      "cow",
                                  "diningtable", "dog",     "horse",  "motorbike",  "person",
                                  "pottedplant", "sheep",   "sofa",   "train",      "tvmonitor"}; // 修改2 25行新增代码，为自训练模型的类别名称
    
for(auto& obj : boxes){
     ...
     auto name    = voclabels[obj.class_label]; // 修改3 100行cocolabels修改为voclabels
	 ...
}

TRT::compile(
    mode,                       // FP32、FP16、INT8
    test_batch_size,            // max batch size
    onnx_file,                  // source 
    model_file,                 // save to
    {},
    int8process,
    "inference",
    "yolov5_trimmed_qat_precision_config_calib.cache" // 修改4 149行，新增校准文件参数
);

```
# 部署
## 环境配置
+ 与ptq相同，修改tensorRT_Pro库中的cmakelist.txt,主要修改TRT、CUDA、cuDNN、opencv、protobuf的路径
```
cmake_minimum_required(VERSION 2.6)
project(pro)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 如果要支持python则设置python路径
set(HAS_PYTHON OFF)                                         # ===== 修改 1 =====
set(PythonRoot "/datav/software/anaconda3")
set(PythonName "python3.9")

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
#set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/usr/local/include/opencv4/")             # ===== 修改 2 =====

set(CUDA_TOOLKIT_ROOT_DIR     "/usr/local/cuda-11.8")       # ===== 修改 3 =====
# set(CUDNN_DIR    "/usr/local/cudnn8.4.0.27-cuda11.6")       # ===== 修改 4 =====
set(TENSORRT_DIR "/opt/TensorRT-8.6.1.6")                   # ===== 修改 5 =====

# set(CUDA_TOOLKIT_ROOT_DIR     "/data/sxai/lean/cuda-10.2")
# set(CUDNN_DIR    "/data/sxai/lean/cudnn7.6.5.32-cuda10.2")
# set(TENSORRT_DIR "/data/sxai/lean/TensorRT-7.0.0.11")

# set(CUDA_TOOLKIT_ROOT_DIR  "/data/sxai/lean/cuda-11.1")
# set(CUDNN_DIR    "/data/sxai/lean/cudnn8.2.2.26")
# set(TENSORRT_DIR "/data/sxai/lean/TensorRT-7.2.1.6")

# 因为protobuf，需要用特定版本，所以这里指定路径
set(PROTOBUF_DIR "/home/protobuf")                   # ===== 修改 6 ======


find_package(CUDA REQUIRED)
find_package(OpenCV)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/application
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib
)

if("${HAS_PYTHON}" STREQUAL "ON")
    message("Usage Python ${PythonRoot}")
    include_directories(${PythonRoot}/include/${PythonName})
    link_directories(${PythonRoot}/lib)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_PYTHON")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(plugin_list SHARED ${cuda_srcs})
target_link_libraries(plugin_list nvinfer nvinfer_plugin)
target_link_libraries(plugin_list cuda cublas cudart cudnn)
target_link_libraries(plugin_list protobuf pthread)
target_link_libraries(plugin_list ${OpenCV_LIBS})

add_executable(pro ${cpp_srcs})

# 如果提示插件找不到，请使用dlopen(xxx.so, NOW)的方式手动加载可以解决插件找不到问题
target_link_libraries(pro nvinfer nvinfer_plugin)
target_link_libraries(pro cuda cublas cudart cudnn)
target_link_libraries(pro protobuf pthread plugin_list)
target_link_libraries(pro ${OpenCV_LIBS})

if("${HAS_PYTHON}" STREQUAL "ON")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/example-python/pytrt)
    add_library(pytrtc SHARED ${cpp_srcs})
    target_link_libraries(pytrtc nvinfer nvinfer_plugin)
    target_link_libraries(pytrtc cuda cublas cudart cudnn)
    target_link_libraries(pytrtc protobuf pthread plugin_list)
    target_link_libraries(pytrtc ${OpenCV_LIBS})
    target_link_libraries(pytrtc "${PythonName}")
    target_link_libraries(pro "${PythonName}")
endif()

add_custom_target(
    yolo
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo
)

add_custom_target(
    yolo_gpuptr
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo_gpuptr
)

add_custom_target(
    yolo_fast
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo_fast
)

add_custom_target(
    centernet
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro centernet
)

add_custom_target(
    alphapose 
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro alphapose
)

add_custom_target(
    retinaface
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro retinaface
)

add_custom_target(
    dbface
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro dbface
)

add_custom_target(
    arcface 
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro arcface
)

add_custom_target(
    bert 
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro bert
)

add_custom_target(
    fall
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro fall_recognize
)

add_custom_target(
    scrfd
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro scrfd
)

add_custom_target(
    lesson
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro lesson
)

add_custom_target(
    pyscrfd
    DEPENDS pytrtc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/example-python
    COMMAND python test_scrfd.py
)

add_custom_target(
    pyinstall
    DEPENDS pytrtc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/example-python
    COMMAND python setup.py install
)

add_custom_target(
    pytorch
    DEPENDS pytrtc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/example-python
    COMMAND python test_torch.py
)

add_custom_target(
    pyyolov5
    DEPENDS pytrtc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/example-python
    COMMAND python test_yolov5.py
)

add_custom_target(
    pycenternet
    DEPENDS pytrtc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/example-python
    COMMAND python test_centernet.py
)

```
+ 与ptq相同，使用trtmodel或者engine进行推理，修改test_yolo_map执行验证
```
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/json.hpp>
#include "app_yolo/yolo.hpp"
#include <vector>
#include <string>

using namespace std;

bool requires(const char* name);

struct BoxLabel{
    int label;
    float cx, cy, width, height;
    float confidence;
};

struct ImageItem{
    string image_file;
    Yolo::BoxArray detections;
};

vector<ImageItem> scan_dataset(const string& images_root){

    vector<ImageItem> output;
    auto image_files = iLogger::find_files(images_root, "*.jpg");

    for(int i = 0; i < image_files.size(); ++i){
        auto& image_file = image_files[i];

        if(!iLogger::exists(image_file)){
            INFOW("Not found: %s", image_file.c_str());
            continue;
        }

        ImageItem item;
        item.image_file = image_file;
        output.emplace_back(item);
    }
    return output;
}

static void inference(vector<ImageItem>& images, int deviceid, const string& engine_file, TRT::Mode mode, Yolo::Type type, const string& model_name){

    auto engine = Yolo::create_infer(
        engine_file, type, deviceid, 0.001f, 0.65f,
        Yolo::NMSMethod::CPU, 10000
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    int nimages = images.size();
    vector<shared_future<Yolo::BoxArray>> image_results(nimages);
    for(int i = 0; i < nimages; ++i){
        if(i % 100 == 0){
            INFO("Commit %d / %d", i+1, nimages);
        }
        image_results[i] = engine->commit(cv::imread(images[i].image_file));
    }
    
    for(int i = 0; i < nimages; ++i)
        images[i].detections = image_results[i].get();
}

void detect_images(vector<ImageItem>& images, Yolo::Type type, TRT::Mode mode, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, type, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", Yolo::type_name(type), mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }
    inference(images, deviceid, model_file, mode, type, name);
}

bool save_to_json(const vector<ImageItem>& images, const string& file){

    Json::Value predictions(Json::arrayValue);
    for(int i = 0; i < images.size(); ++i){
        auto& image = images[i];
        auto file_name = iLogger::file_name(image.image_file, false);
        string image_id = file_name;

        auto& boxes = image.detections;
        for(auto& box : boxes){
            Json::Value jitem;
            jitem["image_id"] = image_id;
            jitem["category_id"] = box.class_label;
            jitem["score"] = box.confidence;

            auto& bbox = jitem["bbox"];
            bbox.append(box.left);
            bbox.append(box.top);
            bbox.append(box.right - box.left);
            bbox.append(box.bottom - box.top);
            predictions.append(jitem);
        }
    }
    return iLogger::save_file(file, predictions.toStyledString());
}

int test_yolo_map(){
    
    /*
    结论：
    1. YoloV5在tensorRT下和pytorch下，只要输入一样，输出的差距最大值是1e-3
    2. YoloV5-6.0的mAP，官方代码跑下来是mAP@.5:.95 = 0.367, mAP@.5 = 0.554，与官方声称的有差距
    3. 这里的tensorRT版本测试的精度为：mAP@.5:.95 = 0.357, mAP@.5 = 0.539，与pytorch结果有差距
    4. cv2.imread与cv::imread，在操作jpeg图像时，在我这里测试读出的图像值不同，最大差距有19。而png图像不会有这个问题
        若想完全一致，请用png图像
    5. 预处理部分，若采用letterbox的方式做预处理，由于tensorRT这里是固定640x640大小，测试采用letterbox并把多余部分
        设置为0. 其推理结果与pytorch相近，但是依旧有差别
    6. 采用warpAffine和letterbox两种方式的预处理结果，在mAP上没有太大变化（小数点后三位差）
    7. mAP差一个点的原因可能在固定分辨率这件事上，还有是pytorch实现的所有细节并非完全加入进来。这些细节可能有没有
        找到的部分
    */

    auto images = scan_dataset("/datasets/datasets/new_dataset/rename/images/calib_test");
    INFO("images.size = %d", images.size());

    string model = "yolov5_trimmed_qat_noqdq";
    detect_images(images, Yolo::Type::V5, TRT::Mode::INT8, model);
    save_to_json(images, model + ".prediction.json");
    return 0;
}


```
+ 准备校准测试集的json
```
import os
import cv2
import json
import logging
import os.path as osp
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)

def process_img(image_filename, data_path, label_path):
    # Open the image file to get its size
    image_path = os.path.join(data_path, image_filename)
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Open the corresponding label file
    label_file = os.path.join(label_path, os.path.splitext(image_filename)[0] + ".txt")
    with open(label_file, "r") as file:
        lines = file.readlines()

    # Process the labels
    labels = []
    for line in lines:
        category, x, y, w, h = map(float, line.strip().split())
        labels.append((category, x, y, w, h))

    return image_filename, {"shape": (height, width), "labels": labels}

def get_img_info(data_path, label_path):
    LOGGER.info(f"Get img info")

    image_filenames = os.listdir(data_path)

    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(partial(process_img, data_path=data_path, label_path=label_path), image_filenames), total=len(image_filenames)))

    img_info = {image_filename: info for image_filename, info in results}
    return img_info


def generate_coco_format_labels(img_info, class_names, save_path):
    # for evaluation with pycocotools
    dataset = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        dataset["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    LOGGER.info(f"Convert to COCO format")
    for i, (img_path, info) in enumerate(tqdm(img_info.items())):
        labels = info["labels"] if info["labels"] else []
        img_id = osp.splitext(osp.basename(img_path))[0]
        img_h, img_w = info["shape"]
        dataset["images"].append(
            {
                "file_name": os.path.basename(img_path),
                "id": img_id,
                "width": img_w,
                "height": img_h,
            }
        )
        if labels:
            for label in labels:
                c, x, y, w, h = label[:5]
                # convert x,y,w,h to x1,y1,x2,y2
                x1 = (x - w / 2) * img_w
                y1 = (y - h / 2) * img_h
                x2 = (x + w / 2) * img_w
                y2 = (y + h / 2) * img_h
                # cls_id starts from 0
                cls_id = int(c)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dataset["annotations"].append(
                    {
                        "area": h * w,
                        "bbox": [x1, y1, w, h],
                        "category_id": cls_id,
                        "id": ann_id,
                        "image_id": img_id,
                        "iscrowd": 0,
                        # mask
                        "segmentation": [],
                    }
                )
                ann_id += 1

    with open(save_path, "w") as f:
        json.dump(dataset, f)
        LOGGER.info(
            f"Convert to COCO format finished. Resutls saved in {save_path}"
        )


if __name__ == "__main__":
    
    # Define the paths
    data_path   = "/home/jarvis/Learn/Datasets/VOC_PTQ/images/val"
    label_path  = "/home/jarvis/Learn/Datasets/VOC_PTQ/labels/val"

    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                   "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                   "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]  # 类别名称请务必与 YOLO 格式的标签对应
    save_path   = "./val.json"

    img_info = get_img_info(data_path, label_path)
    generate_coco_format_labels(img_info, class_names, save_path)

```
+ 在workspace下执行./Pro test_yolo_map,得到结果json，对比计算
```
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Run COCO mAP evaluation
# Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

annotations_path = "val.json"
results_file = "yolov5_trimmed_qat_noqdq.prediction.json"
cocoGt = COCO(annotation_file=annotations_path)
cocoDt = cocoGt.loadRes(results_file)
imgIds = sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

```
# 参考链接
https://blog.csdn.net/qq_40672115/article/details/133774516