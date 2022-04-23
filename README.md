# ONNX Object Localization Network
 Python scripts performing class agnostic object localization using the Object Localization Network model in ONNX.

![!Object-Localization-Network](https://github.com/ibaiGorordo/ONNX-Object-Localization-Network/blob/main/doc/img/output.jpg)
*Original image: https://en.wikipedia.org/wiki/File:Interior_design_865875.jpg*

# Requirements

 * Check the **requirements.txt** file. 
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-Object-Localization-Network.git
cd ONNX-Object-Localization-Network
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

# ONNX model
The original model was converted to ONNX by [PINTO0309](https://github.com/PINTO0309), download the models from the download script in [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/264_object_localization_network) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-Object-Localization-Network/tree/main/models)** folder. 
- The License of the models is Apache-2.0 License: https://github.com/mcahny/object_localization_network/blob/main/LICENSE

# Pytorch model
The original Pytorch model can be found in this repository: https://github.com/mcahny/object_localization_network
 
# Examples

 * **Image inference**:
 ```
 python image_object_localization.py
 ```
 
 * **Webcam inference**:
 ```
 python webcam_object_localization.py
 ```

 * **Video inference**: https://youtu.be/n9qhQJXYUWo
 ```
 python video_object_localization.py
 ```
 ![!Object-Localization-Network video](https://github.com/ibaiGorordo/ONNX-Object-Localization-Network/blob/main/doc/img/oln_box.gif)
  
 *Original video: https://youtu.be/vgJUXvkdS78*

# References:
* Object-Localization-Network model: https://github.com/mcahny/object_localization_network
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* Original paper: https://arxiv.org/abs/2108.06753
