# Segmentation

### Segmentation Example:

### Using Predict Command:
- `python3 predict.py --model /mnt/Data/Model/seg-0000200-NHWC.pb --visualize image --data /mnt/Data/Test_Data/img_001.jpg`

- `python3 predict.py --model /mnt/Data/Model/seg-0000200-NHWC.pb --visualize dataset --data /mnt/Data/Test_Data`

### Model Training:
- `python3 train.py --net SimpleNet --name lights_map --dataset-dir /mnt/Data/Train_Data --save-dir /mnt/Data/Model --epochs 200 --batch-size 16`

### Important Notes:
- Used Python Version: 3.8.10

- Install necessary modules with `sudo pip3 install -r requirements.txt` command.
