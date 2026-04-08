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

# 模型推理，文件夹
python Test_SegNet_Pavements_best.py `
    "D:\Files\Data\IRWSB\analyse\images" `
    "D:\Files\Data\IRWSB\analyse\masks_white" `
    "F:\AAA\2_segnet_best\experiment1\exp_bs_6_last.pth" `
    ".\results_best_pth"


# 模型推理，保存推理结果为红色蒙版风格，并且根据mask真值计算评价指标
# 1. 单张图片推理 + 保存叠加图
python Test_SegNet_Pavements_best_pro.py "D:\Data\test\img_001.jpg" "checkpoints\best.pth" --output ".\results"

# 2. 批量推理文件夹内所有图片 + 保存叠加图 + 计算指标并保存CSV
python Test_SegNet_Pavements_best_pro.py `
    "D:\Files\Data\IRWSB\analyse\images" `
    "F:\AAA\2_segnet_best\experiment1\exp_bs_6_last.pth" `
    --output "D:\Files\GitProject\SegNet_PyTorch-LY\WaterSegment\results_best_pro" `
    --ground_truth "D:\Files\Data\IRWSB\analyse\masks_white_noSuffix" `
    --alpha 0.5

# 3. 仅计算指标不保存图片（不指定--output）
python Test_SegNet_Pavements_best_pro.py "D:\Data\test\images" "checkpoints\best.pth" `
    --ground_truth "D:\Data\test\masks"

# 4. 调整透明度
python Test_SegNet_Pavements_best_pro.py "input.jpg" "model.pth" -o "output" --alpha 0.3

