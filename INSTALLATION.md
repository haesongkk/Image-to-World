#### INSTALL recognize-anything

```
cd third_party
git clone https://github.com/xinyu1205/recognize-anything.git 

cd recognize-anything
py -3.8 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
deactivate

mkdir pretrained
curl.exe -L --progress-bar -o pretrained/ram_plus_swin_large_14m.pth "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"  
```

---

#### INSTALL Grounded-SAM-2


```
cd .\third_party\
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

cd .\Grounded-SAM-2\
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install opencv-python supervision pycocotools
pip install "transformers<=4.54.0"
deactivate

cd .\checkpoints\
bash download_ckpts.sh
```

---

#### INSTALL Hunyuan3D-2

```
cd .\third_party\
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd .\Hunyuan3D-2\ 

py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

(msvc 빌드툴 14.38 버전 설치)

cd .\hy3dgen\texgen\custom_rasterizer\
python setup.py install
cd ../../../
cd .\hy3dgen\texgen\differentiable_renderer\
python setup.py install 
```

---

#### INSTALL ml-depth-pro

```
git clone https://github.com/apple/ml-depth-pro.git

cd ml-depth-pro
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

bash get_pretrained_models.sh

or

mkdir checkpoints
curl.exe -L --progress-bar -o "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" "checkpoints\depth_pro.pt"
```
