# Image-to-World

단일 이미지를 입력으로 받아 객체 단위로 장면을 분해하고, 각 객체의 3D 자산과 깊이 정보를 추정한 뒤 하나의 scene으로 다시 조합하는 modular reconstruction pipeline입니다.

이 프로젝트는 monolithic end-to-end 모델을 바로 학습하는 대신, intermediate artifact를 직접 확인하고 단계별로 개선할 수 있는 구조를 우선 목표로 삼았습니다. 현재 결과물은 metric-accurate reconstruction이라기보다, single-image indoor scene reconstruction을 위한 inspectable prototype에 가깝습니다.

## 프로젝트 개요

### 문제 정의

단일 이미지에서 바로 완성된 3D indoor scene을 복원하는 것은 여전히 어렵습니다. 특히 다음 문제가 한 번에 얽혀 있습니다.

- 어떤 객체가 장면 안에 있는지 식별해야 함
- 객체별 마스크와 가려진 영역을 분리해야 함
- 객체의 대략적인 3D 형상을 복원해야 함
- 상대적 depth를 바탕으로 장면 안의 배치를 추정해야 함
- 최종적으로 하나의 scene mesh로 조합해야 함

이 프로젝트는 이 문제를 7개의 stage로 나누어 풀고 있습니다.

- `extract_tags`
- `generate_masks`
- `complete_objects`
- `generate_meshes`
- `estimate_depth`
- `compose_layout`
- `assemble_scene`

### 입력

- 단일 RGB 이미지
- 기본 입력 경로: `artifacts/raw_image.jpg`

### 출력

- 객체 태그, 마스크, crop, inpaint 결과, depth map, layout JSON, assembled OBJ/MTL
- 최종 출력 예시:
  - `artifacts/assemble_scene/assembled_scene.obj`
  - `artifacts/assemble_scene/assembled_scene.mtl`
  - `artifacts/assemble_scene/assembly_result.json`

### 왜 이런 구조로 만들었는가

이 프로젝트는 처음부터 “좋은 최종 결과”보다 “어디가 잘 되고 어디가 막히는지 분해해서 볼 수 있는 구조”를 우선했습니다.

- stage별 입력/출력이 분리되어 디버깅이 쉬움
- 모델 교체와 실험 반복이 쉬움
- 결과 품질 저하 지점을 추적하기 쉬움
- 포트폴리오 관점에서도 문제 분해와 설계 의도가 분명하게 드러남

## 프로젝트 상태 요약

- End-to-end prototype 연결 완료
- canonical stage 이름과 `artifacts/` 구조 정리 완료
- pipeline orchestrator, manifest, cache, logging 구조 추가 완료
- layout과 scene assembly는 아직 heuristic 성격이 강함
- object completion과 image-to-3D 품질은 계속 개선이 필요한 상태

## 실행 방법

### 1. 입력 이미지 준비

입력 이미지를 아래 경로에 둡니다.

```bash
artifacts/raw_image.jpg
```

기본 설정은 [image_to_world/config.py](image_to_world/config.py)에 모여 있습니다. 입력 경로, 모델 ID, threshold, depth/layout heuristic은 이 파일에서 조정합니다.

### 2. 전체 파이프라인 실행

전체 pipeline을 처음부터 끝까지 실행합니다.

```bash
python run_pipeline.py
```

특정 device를 지정하려면:

```bash
python run_pipeline.py --device cuda
python run_pipeline.py --device cpu
```

이미 생성된 결과를 재사용하면서 없는 stage만 실행하려면:

```bash
python run_pipeline.py --skip-existing
```

기존 결과를 덮어쓰며 다시 생성하려면:

```bash
python run_pipeline.py --overwrite
```

### 3. 특정 구간만 실행

예를 들어 mask 이후부터 assembly까지 다시 돌리고 싶다면:

```bash
python run_pipeline.py --from generate_masks --to assemble_scene
```

depth 이후만 다시 보고 싶다면:

```bash
python run_pipeline.py --from estimate_depth --to assemble_scene
```

### 4. 개별 stage 실행

현재는 루트의 `01_tag.py` 같은 래퍼 파일 대신, stage 모듈을 직접 실행하는 구조입니다.

```bash
python -m image_to_world.stages.extract_tags
python -m image_to_world.stages.generate_masks
python -m image_to_world.stages.complete_objects
python -m image_to_world.stages.generate_meshes
python -m image_to_world.stages.estimate_depth
python -m image_to_world.stages.compose_layout
python -m image_to_world.stages.assemble_scene
```

각 stage도 동일하게 다음 옵션을 지원합니다.

```bash
--device cuda|cpu
--skip-existing
--overwrite
```

### 5. 실행 중 확인할 파일

pipeline 실행 시 아래 파일들을 기준으로 진행 상태를 확인할 수 있습니다.

- `artifacts/manifest.json`: stage별 최신 실행 기록
- `artifacts/logs/pipeline.log`: stage 실행 로그
- stage별 결과 JSON:
  - `artifacts/extract_tags/ram_result.json`
  - `artifacts/generate_masks/result.json`
  - `artifacts/complete_objects/amodal_result.json`
  - `artifacts/generate_meshes/gen3d_result.json`
  - `artifacts/estimate_depth/result.json`
  - `artifacts/compose_layout/scene_layout.json`
  - `artifacts/assemble_scene/assembly_result.json`

### 6. 테스트

경량 테스트는 아래 명령으로 확인할 수 있습니다.

```bash
python -m unittest discover -s tests
```

## 구현 현황


| Stage              | Status    | Inputs                           | Outputs                                                                            | Method                                                                              | Notes                                             |
| ------------------ | --------- | -------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| `extract_tags`     | Done      | `artifacts/raw_image.jpg`        | `artifacts/extract_tags/ram_result.json`                                           | RAM++ 기반 tag extraction, background-like tag filtering                              | 배경성 태그가 섞일 수 있어 후처리 품질이 중요함                       |
| `generate_masks`   | Done      | raw image, extracted tags        | `artifacts/generate_masks/result.json`, masks, crops, overlay                      | Grounding DINO + SAM2                                                               | 상위 stage tag 품질에 영향받음. small object 분리가 불안정할 수 있음 |
| `complete_objects` | Working   | object RGBA crops                | `artifacts/complete_objects/amodal_result.json`, inpaint inputs, masks, amodal RGB | SDXL Inpainting 기반 object completion                                                | 속도 부담이 크고 object category에 따라 품질 편차가 큼            |
| `generate_meshes`  | Working   | amodal RGB images                | `artifacts/generate_meshes/gen3d_result.json`, per-object OBJ/PLY                  | Shap-E 기반 image-to-3D mesh generation                                               | mesh noise, topology 품질, texture 부재가 한계           |
| `estimate_depth`   | Done      | raw image, optional mask JSON    | `artifacts/estimate_depth/result.json`, depth maps, raw depth array                | Depth Anything V2                                                                   | 현재는 relative depth 기반이며 정량 검증은 약함                 |
| `compose_layout`   | Prototype | mask JSON, depth JSON, mesh JSON | `artifacts/compose_layout/scene_layout.json`, layout preview                       | pseudo pinhole camera, relative depth to pseudo-Z mapping, heuristic object scaling | 배치 로직은 초기화 수준. camera, scale, floor prior 보강이 필요  |
| `assemble_scene`   | Prototype | layout JSON, per-object OBJ      | `artifacts/assemble_scene/assembled_scene.obj`, `.mtl`, `assembly_result.json`     | OBJ loading, transform composition, merged scene export                             | 최종 scene consistency와 placement realism은 아직 부족    |


## 다음 작업

### Priority 1

- tag 추출 전 배경 분리
- 객체 배치 정확도 개선
- `complete_objects` stage의 속도 개선 (모델 교체 또는 수치 조정) 
- 텍스처 지원 가능한 3D 생성 모델 검토

### Priority 2

- 시각화 및 디버깅 기반 확보
- depth 품질을 정성 확인을 넘어서 placement 결과와 연결해 검증하기
- transform 계산과 assembly 책임을 더 명확하게 분리하고 calibration 포인트 추가하기
- `generate_meshes` 결과의 noise와 과도한 geometry를 줄이는 후처리 실험 추가

### Priority 3

- scene-level consistency를 위한 후처리 규칙 또는 refinement 단계 고민하기
- background / structural reconstruction까지 포함하는 방향으로 확장하기
- camera recovery, semantic scale prior, scene normalization 도입 검토하기
- 설정 파일 분리, 자동 비교 리포트, 실험 기록 체계 등 유지보수성 강화하기

## 결과물

> `2026-03-27` : 전체 파이프라인 설계 및 각 단계별 기능 구현

#### Input image

![Input](./doc/260327/input.jpg)
이미지 출처: [pixabay simple room image](https://pixabay.com/ko/photos/%ec%9d%b8%ed%85%8c%eb%a6%ac%ec%96%b4-%eb%94%94%ec%9e%90%ec%9d%b8-%ed%98%84%eb%8c%80%ec%a0%81%ec%9d%b8-%ec%8a%a4%ed%83%80%ec%9d%bc-4467768/)

#### Object mask / segmentation overlay

![Mask Overlay](./doc/260327/mask.png)

#### Depth estimation preview

![Depth Preview](./doc/260327/depth.png)

#### Layout preview

![Layout Preview](./doc/260327/layout.png)

#### Scene assembly previews

![Front View](./doc/260327/front.png)
![Side View](./doc/260327/side.png)
![Top View](./doc/260327/top.png)

### Notes

- 배경과 붙어있는 객체는 잘 분리되지 않았다.
- 작거나 경계가 애매한 객체도 잘 분리되지 않았다.
- 객체들의 배치가 원본 이미지와 차이가 크다.
- depth estimation 결과를 3D 점으로 시각화하여 확인하면 좋을 것 같다.
- layout 시각화에 크기와 회전에 대한 시각화를 추가하는게 좋을 것 같다.

## 기술 스택

### Core

- Python
- PyTorch
- NumPy
- Pillow
- OpenCV

### Vision / Segmentation

- RAM++
- Grounding DINO
- SAM2

### Generative Modeling

- Stable Diffusion XL Inpainting
- Shap-E

### Depth / Geometry / Scene Assembly

- Depth Anything V2
- heuristic pseudo-camera projection
- relative depth based object placement
- OBJ / MTL assembly pipeline

### Project Structure

- modular stage pipeline
- dataclass-based config and schema organization
- artifact manifest and cache tracking
- unittest-based lightweight validation

## 참고 자료

- [Zero-Shot Scene Reconstruction from Single Images with Deep Prior Assembly](https://arxiv.org/html/2410.15971v1)
- [Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](https://arxiv.org/html/2411.19492v2)
- [3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework](https://arxiv.org/html/2512.17459v1)
- [DepR: Depth Guided Single-view Scene Reconstruction with Instance-level Diffusion](https://arxiv.org/html/2507.22825v1)
- [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](https://arxiv.org/html/2507.08416v2)
- [PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](https://arxiv.org/html/2603.05888v1)
- [Gen3DSR: Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View](https://arxiv.org/html/2404.03421v2)
- [Open-World Amodal Appearance Completion](https://arxiv.org/html/2411.13019v1)

### 추가 문서

- 좌표계와 pseudo-world 가정은 [doc/coordinate_system.md](doc/coordinate_system.md) 참고

