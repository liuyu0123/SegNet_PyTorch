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

#测试模型
#Powershell 格式
# 先 cd 到 WaterSegment 文件夹
python Test_SegNet_Pavements.py `
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted" `
    "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_gif" `
    ".\weights\checkpoint_pavements_20260309_111733.pth.tar" `
    ".\results"