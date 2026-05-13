# Installation (Submodule Based)

## 0) Initialize Third-Party Submodules

```powershell
git submodule sync --recursive
git submodule update --init --recursive
```

If you need a clean re-download of all third-party repos:

```powershell
git submodule deinit -f --all
git submodule update --init --recursive --force
```

---

## 1) recognize-anything

```powershell
cd third_party/recognize-anything
py -3.8 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
deactivate

mkdir pretrained
curl.exe -L --progress-bar -o pretrained/ram_plus_swin_large_14m.pth "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
```

---

## 2) Grounded-SAM-2

```powershell
cd third_party/Grounded-SAM-2
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install opencv-python supervision pycocotools
pip install "transformers<=4.54.0"
deactivate
```

Checkpoint download:

```bash
cd third_party/Grounded-SAM-2/checkpoints
bash download_ckpts.sh
```

---

## 3) Hunyuan3D-2

```powershell
cd third_party/Hunyuan3D-2
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

cd hy3dgen/texgen/custom_rasterizer
python setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python setup.py install
deactivate
```

---

## 4) ml-depth-pro

```powershell
cd third_party/ml-depth-pro
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .

mkdir checkpoints
curl.exe -L --progress-bar -o checkpoints/depth_pro.pt "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
deactivate
```

---

## 5) PerspectiveFields

```powershell
cd third_party/PerspectiveFields
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$env:PYTHONUTF8="1"
pip install -e .
pip install imageio
deactivate
```

---

## 6) PyTorch3D (WSL Ubuntu)

```bash
cd /mnt/d/Image-to-World/third_party/pytorch3d
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ninja iopath fvcore
pip install -e . --no-build-isolation
```

Quick check:

```bash
python - <<'PY'
import torch
import pytorch3d
print(torch.__version__)
print(torch.cuda.is_available())
print(pytorch3d.__version__)
PY
```

Notes:
- Verified before with `torch==2.5.1+cu121`, `pytorch3d==0.7.9` on WSL.
- If Python version changes, recreate each module `.venv`.
