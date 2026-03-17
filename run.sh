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
python Train_SegNet_Pavements_with_val.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_white `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_white `
    --logs logs/tensorboard  > logs/train_test1.log 2>&1

#测试模型
#Powershell 格式
# 先 cd 到 WaterSegment 文件夹
python Test_SegNet_Pavements.py `
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted" `
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_gif" `
    ".\weights\checkpoint_pavements_20260309_111733.pth.tar" `
    ".\results"