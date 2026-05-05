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
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .

mkdir checkpoints
curl.exe -L --progress-bar -o "checkpoints\depth_pro.pt" "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
```

---

#### INSTALL PerspectiveFields

```
git clone https://github.com/jinlinyi/PerspectiveFields.git

cd .\PerspectiveFields\
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$env:PYTHONUTF8="1"
pip install -e .
pip install imageio
```

---

#### INSTALL PyTorch3D (WSL Ubuntu)

```
cd /mnt/d/Image-to-World/third_party
git clone https://github.com/facebookresearch/pytorch3d.git

cd /mnt/d/Image-to-World/third_party/pytorch3d
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ninja iopath fvcore
pip install -e . --no-build-isolation

python - <<'PY'
import torch
import pytorch3d
print(torch.__version__)
print(torch.cuda.is_available())
print(pytorch3d.__version__)
PY
```

Notes:
- This project was verified with `torch==2.5.1+cu121`, `pytorch3d==0.7.9` on WSL.
- If you need Python 3.10/3.11, install that interpreter first and recreate `.venv`.
