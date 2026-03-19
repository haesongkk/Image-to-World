# Image-to-World

단일 이미지를 입력받아 장면 내 객체를 분리하고, 깊이와 객체별 3D 복원을 활용해 최종적으로 **3D 월드 기반 장면 정보**로 재구성하는 프로젝트입니다. 본 프로젝트는 하나의 거대한 end-to-end 생성 모델을 직접 학습하는 방식보다, **모듈형 파이프라인**을 구성해 각 단계를 독립적으로 검증하고 점진적으로 고도화하는 방향을 채택합니다. 이 접근은 최근 single-image scene reconstruction 연구 중 **Deep Prior Assembly**, **Gen3DSR**, **3D-RE-GEN**과 가장 가깝습니다. 

## 1. 프로젝트 개요

이 프로젝트의 큰 목표는 **단일 이미지 한 장으로부터 장면에 존재하는 객체들의 위치와 형태를 추정하고, 이를 3차원 장면 정보로 변환하는 것**입니다. 다만 현재 단계에서는 PixARMesh 같은 통합 생성 모델을 직접 학습하는 방식보다, **객체 분리 → 깊이 추정 → 객체 보완 → 객체별 3D 생성 → 장면 배치 → scene assembly**로 이어지는 파이프라인을 먼저 구축하는 것이 더 현실적이라고 판단했습니다. Deep Prior Assembly는 zero-shot 조립형 구조를, Gen3DSR는 divide-and-conquer 기반 모듈형 구조를, 3D-RE-GEN은 editable scene을 위한 compositional generation 구조를 제시합니다.

본 프로젝트는 최종적으로 다음과 같은 결과를 목표로 합니다.

- 입력 이미지 내 객체를 인스턴스 단위로 분리
- 장면의 깊이와 기하 정보를 추정
- 가려진 객체를 가능한 범위에서 보완
- 객체별 3D 자산 생성
- 객체를 장면 좌표계에 배치
- 하나의 3D scene으로 조립
- 추후 Unity / Blender / Unreal 등에서 활용 가능한 형태로 출력

## 2. 목표

### 최종 목표

단일 이미지를 입력받아 장면 전체를 **3D 월드 기반 scene representation**으로 재구성하는 시스템을 구현하는 것입니다. 여기에는 객체별 3D 형태, 상대적 배치, 배경 또는 구조물 정보가 포함됩니다. 이 문제는 최근에도 활발히 연구 중이며, Deep Prior Assembly는 zero-shot single-image scene reconstruction을, Gen3DSR는 모듈형 divide-and-conquer reconstruction을, 3D-RE-GEN은 textured object와 background를 포함한 editable reconstruction을 목표로 합니다.

### 현재 목표

초기 단계에서는 하나의 통합 모델을 새로 학습하기보다, **작동 가능한 프로토타입 파이프라인**을 완성하는 것을 목표로 합니다. 즉, 각 단계를 독립적으로 테스트하고, 앞 단계의 출력이 다음 단계의 입력으로 실제 연결되는지를 검증하는 것이 우선입니다. Gen3DSR는 전체 시스템의 end-to-end training 없이 모듈을 결합하는 방식을 강조하고, Deep Prior Assembly도 task-specific data-driven training 없이 priors를 조립하는 방향을 제시합니다.

## 3. 왜 파이프라인 방식으로 접근하는가

본 프로젝트의 원래 큰 목표는 분명히 “단일 이미지 → 3D 월드”입니다. 다만 현재 시점에서 그 목표를 바로 하나의 통합 모델 학습 문제로 가져가면, 프로젝트 전체가 모델 재현과 학습 성패에 과도하게 종속될 수 있습니다. 반면 파이프라인 방식은 각 단계를 독립적으로 제어할 수 있고, 부분 성공도 결과물로 남기기 쉽습니다. 이는 Deep Prior Assembly가 장면 복원을 여러 하위 문제로 분해하고, Gen3DSR가 scene-level understanding과 object-level reconstruction을 나누며, 3D-RE-GEN이 asset detection, reconstruction, placement를 조합하는 구조와도 맞닿아 있습니다.

즉, 본 프로젝트는 다음과 같은 철학을 가집니다.

- 문제를 한 번에 풀지 않고 하위 문제로 나눈다.
- 각 하위 문제에 강한 모델이나 기술을 배치한다.
- 전체를 모듈형으로 설계해 교체 가능하게 만든다.
- 최종적으로는 더 통합된 scene generation 방향으로 확장 가능성을 열어둔다.

## 4. 전체 파이프라인

본 프로젝트의 기본 파이프라인은 다음과 같습니다.

**입력 이미지 → 객체 / 배경 분리 → depth 추정 → 객체별 가려진 부분 보완 → 객체별 3D 생성 → 카메라 / ground / depth 기반 배치 → 최종 scene assembly**

이 흐름은 Deep Prior Assembly의 zero-shot scene reconstruction 파이프라인, Gen3DSR의 scene-first / object-second divide-and-conquer 구조, 3D-RE-GEN의 object + background reconstruction 구조를 종합한 방향입니다. Deep Prior Assembly는 Grounded-SAM, Stable Diffusion, OpenCLIP, ShapE, Omnidata를 조합하고, Gen3DSR는 depth·semantic 정보 추출 후 object reconstruction과 scene composition을 수행하며, 3D-RE-GEN은 instance segmentation, context-aware inpainting, 2D-to-3D asset creation, constrained optimization을 결합합니다. 

## 5. 구현 예정 모듈

### 5-1. 객체 / 배경 분리 모듈

이 단계의 목표는 입력 이미지에서 **객체 인스턴스 마스크**와 **배경 영역**을 안정적으로 얻는 것입니다.
주요 후보로는 **SAM 2**, **Grounding DINO 1.5**, **Grounded SAM 2**, **BiRefNet**을 고려합니다. SAM 2는 promptable segmentation foundation model이며, Grounding DINO 1.5는 open-set / open-world object detection 계열의 강력한 후보입니다. Grounded SAM 2는 Grounding DINO와 SAM 2를 결합해 open-vocabulary detection과 segmentation을 함께 수행할 수 있습니다. 배경 분리나 고해상도 foreground extraction 쪽에서는 BiRefNet도 유력한 후보입니다. 

**우선 후보**

- Grounded SAM 2
- Grounding DINO 1.5 + SAM 2
- BiRefNet

### 5-2. Depth 추정 모듈

이 단계의 목표는 장면의 앞뒤 관계, 상대적 거리감, 배치 기준이 되는 깊이 정보를 얻는 것입니다.
우선 후보로는 **Depth Pro**, **Depth Anything V2**, **Omnidata**를 둡니다. Depth Pro는 zero-shot metric monocular depth estimation을 표방하며, absolute-scale depth를 intrinsics 없이 예측할 수 있다고 보고합니다. Depth Anything V2는 V1 대비 더 세밀하고 robust한 depth를 제공하는 최근 강한 후보입니다. Omnidata는 Deep Prior Assembly에서 실제로 사용된 기준선 성격의 depth prior입니다.

**우선 후보**

- Depth Pro
- Depth Anything V2
- Omnidata

### 5-3. 카메라 / 기하 추정 모듈

이 단계의 목표는 객체를 월드 좌표계에 놓기 위한 **camera parameters**, **ground plane**, **scene geometry hints**를 얻는 것입니다. Gen3DSR는 scene을 holistic하게 먼저 분석해 depth와 semantic 정보를 얻은 뒤 object reconstruction으로 넘어가고, 3D-RE-GEN은 precise camera recovery와 estimated ground plane에 기반한 4-DoF differentiable optimization을 사용합니다. 따라서 본 프로젝트도 단순 depth map만이 아니라, **camera / ground 추정**을 scene assembly의 핵심 요소로 다룰 예정입니다.

**초기 구현 방향**

- pinhole camera + depth 기반 간단한 배치
- ground plane 추정 기반 정렬
- 이후 differentiable alignment로 확장

### 5-4. 객체별 가려진 부분 보완 / Amodal Completion 모듈

이 단계의 목표는 부분적으로만 보이는 객체를 **더 완전한 2D 입력**으로 보완하여 이후 3D 생성 품질을 높이는 것입니다. Deep Prior Assembly는 Stable Diffusion과 OpenCLIP 기반 filtering을 사용하고, Gen3DSR는 partially occluded object completion을 포함하며, 3D-RE-GEN은 context-aware generative inpainting을 핵심 단계로 둡니다.

실험 후보로는 **SDXL Inpainting**, **Stable Diffusion Inpainting**, **LaMa**를 우선 고려합니다. Diffusers 문서는 Stable Diffusion Inpainting과 SDXL Inpainting을 대표적 inpainting 계열로 제시하며, LaMa는 large-mask inpainting에서 여전히 강한 baseline입니다.

**우선 후보**

- SDXL Inpainting
- Stable Diffusion Inpainting
- LaMa

### 5-5. 객체별 2D → 3D 생성 모듈

이 단계의 목표는 객체 crop 또는 보완된 객체 이미지를 입력으로 받아 **3D object asset**을 생성하는 것입니다. Deep Prior Assembly는 이 단계에서 **ShapE**를 사용했고, 3D-RE-GEN은 보다 일반적인 2D-to-3D asset creation을 포함합니다.

최근 강한 후보로는 **TRELLIS**, **Hunyuan3D-2**, 그리고 기준선으로 **ShapE**를 고려합니다. TRELLIS는 text 또는 image prompt를 받아 meshes를 포함한 다양한 3D 자산 포맷을 생성하는 large 3D asset generation model입니다. Hunyuan3D-2는 bare mesh 생성 후 texture map을 합성하는 2-stage pipeline을 채택하고 있어 파이프라인형 프로젝트와 잘 맞습니다.

**우선 후보**

- TRELLIS
- Hunyuan3D-2
- Baseline: ShapE

### 5-6. 객체 정렬 / 월드 좌표 배치 모듈

이 단계의 목표는 생성된 객체를 원본 이미지 장면 기준의 **위치, 방향, 크기**에 맞게 배치하는 것입니다. Deep Prior Assembly는 location, orientation, scale을 최적화해 장면에 배치하고, Gen3DSR는 monocular depth guides를 사용해 scene composition을 수행하며, 3D-RE-GEN은 ground plane에 정렬되는 4-DoF differentiable optimization을 사용합니다.

**구현 방향**

- depth-guided placement
- ground-plane alignment
- 필요 시 differentiable rendering / optimization 적용
- collision 완화와 물리적으로 어색한 배치 감소

### 5-7. 배경 / 구조물 복원 모듈

이 단계의 목표는 객체들만 따로 있는 상태가 아니라, 장면을 장면답게 보이게 하는 **배경과 구조물**을 함께 복원하는 것입니다. 3D-RE-GEN은 개별 객체뿐 아니라 **reconstructed background**를 함께 생성해 object placement를 공간적으로 제약하고, lighting과 simulation의 기반으로 활용한다고 설명합니다.

초기 버전에서는 이 모듈을 단순화하여:

- floor / wall plane만 추정하거나
- 배경을 별도 구조물 mesh로 처리하거나
- 배경을 scene anchor 용도로만 사용하는 방식
으로 시작할 수 있습니다.

### 5-8. 최종 Scene Assembly / 출력 포맷 모듈

최종 단계의 목표는 객체와 배경을 하나의 **3D scene representation**으로 묶고, 이후 다른 툴에서 사용할 수 있도록 정리하는 것입니다. Gen3DSR는 triangle mesh 기반 scene components를, 3D-RE-GEN은 editable, modifiable scene을 강조합니다. 따라서 본 프로젝트도 최종 출력은 **mesh 중심**으로 두되, 초기에는 객체별 mesh + metadata(JSON) 조합으로 시작하고 이후 textured scene 형태로 확장할 계획입니다.

## 6. 관련 연구

### 핵심 참고 논문

- **Zero-Shot Scene Reconstruction from Single Images with Deep Prior Assembly**
단일 이미지 장면 복원을 여러 하위 문제로 분해하고, Grounded-SAM, Stable Diffusion, OpenCLIP, ShapE, Omnidata 같은 deep priors를 zero-shot manner로 조립하는 연구입니다. 본 프로젝트의 **핵심 철학**에 가장 가깝습니다. 
- **Gen3DSR: Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View**
scene을 holistic하게 먼저 보고 depth와 semantic 정보를 얻은 뒤, object-level reconstruction을 수행하고 다시 scene으로 조립하는 divide-and-conquer 파이프라인입니다. 본 프로젝트의 **전체 구조 설계도**에 가장 가깝습니다.
- **3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework**
single image를 textured 3D objects와 background로 재구성하고, instance segmentation, context-aware generative inpainting, 2D-to-3D asset creation, constrained optimization을 통해 editable scene을 만드는 compositional framework입니다. 본 프로젝트의 **실전 구현 감각**과 **결과물 지향성**에 가장 가깝습니다.

### 부분 참고 논문

- **Diorama**
CAD retrieval과 architecture-aware scene modeling 관점에서 참고할 가치가 있지만, 본 프로젝트는 기존 CAD를 검색해 조립하는 방향보다 객체를 직접 생성하는 방향에 더 가깝기 때문에 핵심 축으로 두지는 않습니다.
- **DepR**
depth-guided object reconstruction과 instance-level diffusion 관점에서 객체별 3D 생성 모듈을 고도화할 때 참고할 수 있습니다. 다만 프로젝트 전체 구조보다는 **객체 복원 모듈의 강화**에 더 가깝습니다.

### 방향성 참고

- **PixARMesh**
단일 이미지에서 실내 scene mesh를 unified autoregressive model로 직접 생성하는 흥미로운 방향입니다. 다만 본 프로젝트 초기 단계에서는 재현·학습·디버깅 난도가 높아, 현재는 메인 구현축보다는 **장기적 비전**으로만 참고합니다.

## 7. 현재 진행 상황

현재까지 다음 내용을 중심으로 조사 및 방향 정리를 진행했습니다.

- 단일 이미지 기반 3D scene reconstruction 관련 최근 논문 조사
- 파이프라인형 접근과 통합 생성형 접근 비교
- 핵심 참고 논문 3개 선정
  - Deep Prior Assembly
  - Gen3DSR
  - 3D-RE-GEN
- 객체 분리, depth, inpainting, 2D→3D asset generation 후보 기술 조사
- 프로젝트를 end-to-end 학습형보다 **모듈형 파이프라인**으로 진행하기로 방향 정리

## 8. 향후 계획

### 단기 계획

- README와 프로젝트 구조 정리
- 최소 실행 파이프라인 설계
- 객체 분리와 depth 추정부터 먼저 테스트
- 객체 하나를 3D로 생성한 뒤 원래 장면 위치에 다시 배치하는 최소 데모 구현

### 중기 계획

- 객체별 가려진 부분 보완 추가
- 여러 객체에 대한 반복 적용
- 배경 / 구조물 복원 추가
- scene assembly 결과 시각화
- 출력 포맷 정리 및 외부 툴 연동 가능성 검토

### 장기 계획

- scene quality 고도화
- textured asset generation 강화
- geometry / placement optimization 개선
- 필요 시 더 통합된 scene generation 계열 연구 방향 흡수

## 9. 프로젝트 방향 한 줄 요약

본 프로젝트는 단일 이미지를 입력으로 받아 객체를 분리하고, 깊이와 기하 정보를 추정하며, 객체별 3D 자산을 생성한 뒤 이를 장면 좌표계에 배치하여 최종적으로 하나의 3D 월드 기반 scene으로 재구성하는 **모듈형 파이프라인 시스템**을 목표로 한다. 그 구현 철학은 Deep Prior Assembly, Gen3DSR, 3D-RE-GEN 계열의 최근 연구 흐름을 따른다.