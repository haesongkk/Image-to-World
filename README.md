# Image-to-World

**Image-to-World**는 단일 2D 이미지로부터 3D scene을 구성하는 실험적 파이프라인 프로젝트이다.  
입력 이미지에서 객체를 분리하고, 각 객체를 3D mesh로 생성한 뒤, depth map과 카메라 정보를 기반으로 객체의 위치와 크기를 추정하여 하나의 3D world로 조립한다.

이 프로젝트는 하나의 이미지를 단순히 3D 모델 하나로 변환하는 것이 아니라, 이미지 속 여러 객체를 개별 asset으로 분리한 뒤 다시 3D 공간에 배치하는 구조를 사용한다. 이를 통해 객체 단위의 3D 생성, point cloud 기반 위치 추정, GLB scene assembly 과정을 하나의 파이프라인으로 연결하였다.

전체 과정은 `Segmentation → Generation → Placement` 단계로 구성된다.  
`Segmentation` 단계에서는 배경 제거, 객체 태그 추출, 객체별 마스크 생성을 수행한다.  
`Generation` 단계에서는 객체별 crop 이미지를 기반으로 3D mesh를 생성하고 remesh를 수행한다.  
`Placement` 단계에서는 depth와 카메라 파라미터를 이용해 객체별 3D 위치를 추정하고, 최종 GLB scene을 생성한다.

현재 프로젝트는 단일 이미지 기반 3D scene reconstruction을 객체 단위 파이프라인으로 구현한 프로토타입이며, 각 단계의 결과를 분리해 확인하고 개선할 수 있도록 구성하였다. 향후 객체 배치 정밀도 향상, occlusion 복원, texture baking, 배경 복원 등을 추가하여 장면 복원 품질을 개선할 예정이다.

## 결과물

### 입력 이미지

![원본 이미지](./doc/260505/raw_image.jpg)

### 출력 GLB 스크린샷

![블렌더 스크린샷](./doc/260505/screenshot.png)

### 개발 기록

[`2026-04-24` : 3D 메쉬 생성 모델 변경](./doc/260424/260424.md)

[`2026-04-10` : 객체 배치 정확도 개선](./doc/260409/260409.md)

[`2026-03-27` : 전체 파이프라인 설계 및 각 단계별 기능 구현](./doc/260327/260327.md)

## 파이프라인

### SEGMENTATION

#### 이미지 배경 제거

![배경 제거 이미지](./doc/260505/raw_image_birefnet.png)

입력 이미지에서 주요 객체를 더 안정적으로 분리하기 위해 `BiRefNet`을 사용하여 배경을 제거한다.  
이 단계는 이후 객체 태그 추출과 마스크 생성 과정에서 배경 영역이 불필요한 객체로 인식되는 것을 줄이기 위한 전처리 단계이다.

#### 객체 태그 추출

```text
--------------
pretrained/ram_plus_swin_large_14m.pth
--------------
load checkpoint from pretrained/ram_plus_swin_large_14m.pth
vit: swin_l
Image Tags:  alcohol | appliance | beverage | black | blender | bottle | coffee machine | liquor | wine | home appliance | kitchenware | lid | liquid | mixer | olive | olive oil | wine bottle
图像标签:  酒精  | 设备  | 饮料  | 黑色 | 搅拌机  | 瓶  | 咖啡机 | 酒 | 葡萄酒 | 家用电器 | 厨房用具 | 盖子  | 液体  | 搅拌机 | 橄榄  | 橄榄油  | 酒瓶
```

배경이 제거된 이미지를 `Recognize Anything(RAM++)` 모델에 입력하여 이미지 안에 존재할 가능성이 높은 객체 태그를 추출한다.
이 단계에서 얻은 태그는 다음 단계의 `Grounded-SAM-2` 입력 프롬프트로 사용되며, 객체별 마스크를 생성하기 위한 후보 객체 목록 역할을 한다.

#### 객체별 마스크 추출

![객체 마스크 이미지](./doc/260505/grounded_sam2_annotated_image_with_mask.jpg)

추출된 객체 태그를 `Grounded-SAM-2`에 입력하여 이미지 내 객체 위치와 영역을 검출한다.
검출 결과는 객체별 segmentation mask로 저장되며, 각 마스크는 이후 객체 크롭, 3D mesh 생성, depth 기반 위치 추정에 공통으로 사용된다.

---

### GENERATION

#### 객체별 3D mesh 생성

![어셋 생성 결과 이미지](./doc/260505/image%20%284%29.png)

객체별 마스크를 기준으로 원본 이미지에서 각 객체 영역을 crop한 뒤, `Hunyuan3D-2`에 입력하여 객체 단위의 3D mesh를 생성한다.
전체 장면을 한 번에 복원하는 방식이 아니라, 이미지 속 객체를 개별 asset으로 분리하여 생성한 뒤 후속 단계에서 다시 배치하는 구조이다.

#### Remesh

![리메시 결과 이미지](./doc/260505/image%20%285%29.png)

`Hunyuan3D-2`로 생성된 mesh는 형태는 확인 가능하지만 폴리곤 수가 많고 구조가 정리되지 않은 경우가 있다.
따라서 `pymeshlab` 기반 remesh 과정을 통해 mesh의 face 수를 줄이고, 이후 장면 조립 및 렌더링에 사용하기 쉬운 형태로 경량화한다.
현재는 geometry를 줄이는 단계에 초점을 두고 있으며, remesh 이후 텍스처가 손상될 수 있기 때문에 향후 texture baking 과정이 필요하다.

---

### PLACEMENT

#### Depth 추정

![Depth map 이미지](./doc/260505/raw_image%20copy.jpg)

`ml-depth-pro`를 사용하여 입력 이미지의 depth map을 추정한다.
이 depth map은 각 픽셀이 카메라로부터 어느 정도 떨어져 있는지를 나타내며, 2D 이미지 좌표를 3D 공간 좌표로 역투영하기 위한 핵심 정보로 사용된다.

#### 카메라 파라미터 및 자세 추정 기반 Point Cloud 생성

![Point Cloud 시각화 이미지](./doc/260505/pointcloud_4views.png)

`PerspectiveFields`를 통해 카메라의 초점 거리, 중심점, roll/pitch, FOV 등의 시점 정보를 추정한다.
이후 depth map, 객체 마스크, 카메라 파라미터를 결합하여 각 객체 영역의 픽셀을 3D point cloud로 변환한다.
즉, 2D 이미지에서 분리된 객체 영역을 실제 3D 공간상의 점 집합으로 바꾸어 객체별 위치와 크기를 계산할 수 있는 중간 표현을 만든다.

#### 초기 배치

![초기 배치 시각화 이미지](./doc/260505/raw_transform_4views.png)

객체별 point cloud의 중심 위치와 공간 범위를 계산하여 각 3D mesh의 초기 transform을 생성한다.
translation은 point cloud의 중심값을 기준으로 설정하고, scale은 point cloud가 차지하는 3D bounding box 크기를 기준으로 설정한다.
생성된 초기 transform은 최종 GLB 장면 조립 단계에서 각 객체 mesh를 배치하는 기준값으로 사용된다.

#### Differential Rendering

![Differential Rendering 과정 이미지](./doc/260505/render_step_0000_loss_2.313139.png)

초기 배치만으로는 객체의 위치, 크기, 회전이 원본 이미지와 정확히 일치하지 않을 수 있다.
이를 보정하기 위해 differentiable rendering을 사용하여 현재 3D 배치 결과를 다시 2D 이미지로 렌더링하고, 원본 이미지 또는 목표 실루엣과의 차이가 줄어들도록 객체 transform을 반복적으로 최적화한다.
현재 프로젝트에서는 초기 배치 결과를 더 정밀하게 맞추기 위한 실험적 보정 단계로 구성되어 있다.

## 기술 스택

### Language / Environment

| 기술 | 사용 목적 |
| --- | --- |
| **Python 3.11** | 전체 파이프라인 제어, 외부 모델 실행, 중간 결과 후처리, 최종 3D scene 조립 |
| **Windows** | 프로젝트 개발 및 실행 환경 |
| **CUDA GPU** | Hunyuan3D-2, Grounded-SAM-2, DepthPro 등 딥러닝 모델 추론 가속 |

---

### AI / Computer Vision Models

| 기술 | 사용 목적 |
| --- | --- |
| **BiRefNet** | 입력 이미지의 배경 제거 |
| **Recognize Anything(RAM++)** | 이미지 내 객체 후보 태그 추출 |
| **Grounded-SAM-2** | 태그 기반 객체 검출 및 객체별 segmentation mask 생성 |
| **Hunyuan3D-2** | 객체 crop 이미지를 기반으로 객체별 3D mesh 생성 |
| **ml-depth-pro** | 단일 이미지의 depth map 추정 |
| **PerspectiveFields** | 카메라 파라미터, FOV, roll/pitch 등 시점 정보 추정 |

---

### 3D Processing

| 기술 | 사용 목적 |
| --- | --- |
| **Trimesh** | GLB/OBJ mesh 로드, 변환, 저장 및 scene 조립 |
| **PyMeshLab** | 생성된 mesh의 polygon 수를 줄이기 위한 remesh/decimation 처리 |
| **Point Cloud 기반 위치 추정** | depth map, camera parameter, object mask를 결합하여 객체별 3D 위치와 크기 추정 |
| **GLB Export** | 객체별 mesh와 transform을 결합한 최종 3D scene 저장 |

---

### Rendering / Optimization

| 기술 | 사용 목적 |
| --- | --- |
| **PyTorch3D** | Differentiable Rendering 기반 객체 transform 보정 실험 |
| **Differentiable Rendering** | 초기 배치된 3D 객체를 다시 2D로 렌더링하고, 원본 이미지와의 차이를 줄이는 방향으로 위치/크기/회전 최적화 |

> 현재 `Differentiable Rendering` 단계는 기본 파이프라인에서는 주석 처리되어 있으며, 초기 배치 결과를 정밀하게 보정하기 위한 실험 단계로 구성되어 있다.

---

### Data Processing / Utility

| 기술 | 사용 목적 |
| --- | --- |
| **NumPy** | depth, mask, point cloud, transform 계산 |
| **OpenCV** | 이미지 로드, 마스크 처리, 시각화 이미지 생성 |
| **Pillow** | 이미지 입출력 및 crop 처리 |
| **JSON** | 객체별 transform, camera parameter, 중간 결과 데이터 저장 |
| **Pathlib** | 프로젝트 내부 경로 관리 |

## 실행 방법

### 1. 리포지토리 클론

```bash
git clone https://github.com/haesongkk/Image-to-World.git
cd Image-to-World
```

### 2. 외부 모델 및 가중치 준비

프로젝트 루트에 `third_party` 폴더를 만들고, 각 단계에서 사용하는 외부 모델을 설치한다.

필요한 외부 리포지토리는 다음과 같다.

| 단계           | 필요 모델 / 리포지토리        | 사용 목적                                       |
| ------------ | -------------------- | ------------------------------------------- |
| Segmentation | `BiRefNet`           | 입력 이미지 배경 제거                                |
| Segmentation | `recognize-anything` | 이미지 내 객체 태그 추출                              |
| Segmentation | `Grounded-SAM-2`     | 객체별 마스크 생성                                  |
| Generation   | `Hunyuan3D-2`        | 객체별 3D mesh 생성                              |
| Placement    | `ml-depth-pro`       | depth map 추정                                |
| Placement    | `PerspectiveFields`  | 카메라 파라미터 및 자세 추정                            |
| Optimization | `PyTorch3D`          | differentiable rendering 기반 transform 보정 실험 |

각 모델은 별도의 가상환경과 체크포인트를 필요로 하므로, 설치 명령은 `INSTALLATION.md`를 기준으로 진행한다.

### 3. 입력 이미지 준비

실행할 이미지를 프로젝트 루트의 `data` 폴더 안에 `raw_image.jpg` 이름으로 배치한다.

```text
Image-to-World/
├─ data/
│  └─ raw_image.jpg
├─ src/
├─ third_party/
└─ run.py
```

현재 파이프라인 코드는 입력 이미지 경로를 다음과 같이 고정해서 사용한다.

```python
image_path = project_root / "data" / "raw_image.jpg"
```

따라서 다른 이미지를 사용하려면 파일명을 `raw_image.jpg`로 맞추거나, 각 stage 코드의 `image_path` 값을 수정해야 한다.

### 4. 전체 파이프라인 실행

프로젝트 루트에서 다음 명령을 실행한다.

```bash
python run.py
```

`run.py`는 내부적으로 다음 순서로 전체 파이프라인을 실행한다.

```text
1. Segmentation
   - BiRefNet 배경 제거
   - RAM++ 객체 태그 추출
   - Grounded-SAM-2 객체 마스크 생성
   - 객체별 mask/crop 생성

2. Generation
   - crop 이미지별 Hunyuan3D-2 mesh 생성
   - 생성 mesh remesh 처리

3. Placement
   - DepthPro depth map 추정
   - PerspectiveFields 카메라 정보 추정
   - 객체별 point cloud 생성
   - 초기 transform 계산
   - 최종 scene GLB 조립
```

### 5. 단계별 결과 확인

실행 결과는 `output` 폴더 아래에 단계별로 저장된다.

```text
output/
├─ BiRefNet/
│  └─ 배경 제거 결과
├─ Grounded-SAM-2/
│  ├─ 객체 검출 및 마스크 결과
│  └─ crops/
│     └─ 객체별 crop 이미지
├─ Hunyuan3D-2/
│  └─ 객체별 3D mesh 결과
├─ remesh/
│  └─ remesh 처리된 mesh 결과
├─ ml-depth-pro/
│  └─ depth map 결과
├─ PerspectiveFields/
│  └─ 카메라 파라미터 및 자세 추정 결과
├─ pointcloud/
│  └─ 객체별 point cloud 결과
├─ transform/
│  └─ 객체별 초기 transform 결과
└─ scene/
   └─ 최종 GLB scene 결과
```

### 6. 최종 결과 확인

최종적으로 생성된 GLB 파일은 `output/scene` 경로에서 확인할 수 있다.

생성된 GLB 파일은 Blender, Unity, Unreal Engine 등 GLB 형식을 지원하는 3D 툴에서 열어 확인할 수 있다.


## 참고 자료

- [Zero-Shot Scene Reconstruction from Single Images with Deep Prior Assembly](https://arxiv.org/html/2410.15971v1)
- [Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](https://arxiv.org/html/2411.19492v2)
- [3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework](https://arxiv.org/html/2512.17459v1)
- [DepR: Depth Guided Single-view Scene Reconstruction with Instance-level Diffusion](https://arxiv.org/html/2507.22825v1)
- [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](https://arxiv.org/html/2507.08416v2)
- [PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](https://arxiv.org/html/2603.05888v1)
- [Gen3DSR: Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View](https://arxiv.org/html/2404.03421v2)
- [TEASER: Fast and Certifiable Point Cloud Registration](https://arxiv.org/abs/2001.07715)
- [Modular Primitives for High-Performance Differentiable Rendering](https://github.com/NVlabs/nvdiffrast.git)
