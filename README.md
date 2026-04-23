# Image-to-World

단일 이미지를 입력으로 받아 객체 단위로 장면을 분해하고, 각 객체의 3D 자산과 깊이 정보를 추정한 뒤 하나의 scene으로 다시 조합하는 modular reconstruction pipeline입니다.

## 프로젝트 개요

### 문제 정의

단일 이미지에서 바로 완성된 3D indoor scene을 복원하는 것은 여전히 어렵습니다. 특히 다음 문제가 한 번에 얽혀 있습니다.

- 어떤 객체가 장면 안에 있는지 식별해야 함
- 객체별 마스크와 가려진 영역을 분리해야 함
- 객체의 대략적인 3D 형상을 복원해야 함
- 상대적 depth를 바탕으로 장면 안의 배치를 추정해야 함
- 최종적으로 하나의 scene mesh로 조합해야 함

이 프로젝트는 이 문제를 여러 단계로 나누어 풀고 있습니다.

## 개발 환경

- 언어: Python 3.11
- 실행 환경: 
- 외부 의존 리포지토리: 
- 필수 가중치/체크포인트:

## 실행 방법

### 전체 파이프라인 실행

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

### 개별 stage 실행


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
## 결과물

[`2026-04-24` : 3D 메쉬 생성 모델 변경](./doc/260424/260424.md)

[`2026-04-10` : 객체 배치 정확도 개선](./doc/260409/260409.md)

[`2026-03-27` : 전체 파이프라인 설계 및 각 단계별 기능 구현](./doc/260327/260327.md)


## 기술 스택



## 참고 자료

- [Zero-Shot Scene Reconstruction from Single Images with Deep Prior Assembly](https://arxiv.org/html/2410.15971v1)
- [Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](https://arxiv.org/html/2411.19492v2)
- [3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework](https://arxiv.org/html/2512.17459v1)
- [DepR: Depth Guided Single-view Scene Reconstruction with Instance-level Diffusion](https://arxiv.org/html/2507.22825v1)
- [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](https://arxiv.org/html/2507.08416v2)
- [PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](https://arxiv.org/html/2603.05888v1)
- [Gen3DSR: Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View](https://arxiv.org/html/2404.03421v2)
- [Open-World Amodal Appearance Completion](https://arxiv.org/html/2411.13019v1)
- [TEASER: Fast and Certifiable Point Cloud Registration](https://arxiv.org/abs/2001.07715)