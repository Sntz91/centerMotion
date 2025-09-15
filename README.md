# CenterMotion

# 1. Data
- You must first use `python -m data.extract_frames.py` to load the data 
  into `/inputs/train` and `/inputs/val`. Note that you must specify the 
  mp4 input video path and the xml file (in cvat format) within the 
  main function call from `data/extract_frames.py`. 
  The generated data within each folder looks as follows:
  - `frame_xxxxx.jpg`: The image file.
  - `frame_xxxxx.txt`: The corresponding file with the ground truth center points. 
  - `frame_xxxxx_boxes.txt`: The corresponding file with the ground truth bounding boxes. (necessary to determine object size for evaluation)
- Maybe you need to do extract frames twice because its buggy.
- You can check the transformations with `python -m data.transformations`
- You can check the whole input data for the model with `python -m data.dataset`

# Training
- with `python train.py` you can start training according to `config.yaml`. 
  This file creates a folder within `experiments` according to the name given 
  in the config. To prevent unintended overwriting the file won't overwrite 
  a folder. 
- with `python evaluation.py` you can evaluate your model, the results will 
  be saved within `experiments/[name]/evaluation/`.
- You can use tensorboard to monitor the training process: 
  `tensorboard --logdir=experiments/[name]/logs --host=0.0.0.0` 
  (you need the host part only when you are remote)
  
  
# Interference 
- Files for interference are currently lokated within `utils` but are 
  currently refactored and will be consolidated within the `inference.py` 
  file.
