cd WaterSegment
#训练模型，不指定输出结果模型名称
python Train_SegNet_Pavements.py \
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted" \
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_gif" \
    ./logs
#Powershell 格式
python Train_SegNet_Pavements.py `
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted" `
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_gif" `
    "./logs"

#训练模型，指定输出结果模型名称
python Train_SegNet_Pavements.py \
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted" \
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_gif" \
    ./logs \
    --weight-fn weights/water_seg.pth.tar

#训练模型（指定train和val路径）
python train_water.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_white `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_white `
    --logs logs/tensorboard  > logs/train_test1.log 2>&1

#训练模型（无需依赖json配置文件）
python train_water_nojson.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_white `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_white `
    --epochs 1 `
    --batch-size 4 `
    --learning-rate 5e-4 `
    --model-dir checkpoints/experiment1 `
    --model-name experiment1 `
    --log-dir logs/experiment1 `
    --log-name experiment1 `
    --save-interval 0
#完整参数训练
python train_water_nojson.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_white `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_white `
    --epochs 50 `
    --batch-size 4 `
    --learning-rate 1e-3 `
    --sgd-momentum 0.9 `
    --bn-momentum 0.5 `
    --loss-weights "1.0,1.0" `
    --model-dir checkpoints/exp_full_params `
    --model-name segnet_exp1 `
    --log-dir logs/exp_full_params `
    --log-name segnet_exp1_log `
    --save-interval 10


#测试模型（测试集评估性能）
# 基础测试
python test_water.py `
    --model .\weights\checkpoint_pavements_20260309_111733.pth.tar `
    --images D:\Files\Data\IRWSB\test\images `
    --masks D:\Files\Data\IRWSB\test\masks_white

# 指定输出路径
python test_water.py `
    --model .\weights\checkpoint_pavements_20260309_111733.pth.tar `
    --images D:\Files\Data\IRWSB\test\images `
    --masks D:\Files\Data\IRWSB\test\masks_white `
    --output results/segnet_test.csv

#测试模型(保存推理结果)
#Powershell 格式
# 先 cd 到 WaterSegment 文件夹
python Test_SegNet_Pavements.py `
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted" `
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_gif" `
    ".\weights\checkpoint_pavements_20260309_111733.pth.tar" `
    ".\results"