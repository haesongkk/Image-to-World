# Image-to-World 프로젝트 설명 문서

## 1. 프로젝트 개요

### 프로젝트명
**Image-to-World**

### 한 줄 설명
단일 입력 이미지로부터 장면 내의 구별 가능한 객체들을 인식하고, 각 객체의 공간적 구조와 시각적 속성을 추정하여, 이를 **고수준 그래픽 표현이 가능한 3차원 장면 데이터**로 변환하고 시각화하며, 추가로 **Unity, Blender, Unreal 같은 3D 툴에서도 활용 가능한 형식으로 저장**하는 프로젝트이다.

### 프로젝트 소개
이 프로젝트의 목표는 단순히 이미지 속 객체를 탐지하는 것을 넘어서, 이미지 한 장을 분석하여 장면을 구성하는 객체들이 **무엇인지**, **어디에 있는지**, **어떤 형태를 가지는지**, 그리고 **어떤 시각적·재질적 특성을 가지는지**를 추정하고, 이를 3차원 공간상의 구조화된 장면 정보로 바꾸는 것이다.

즉, 최종적으로는 입력 이미지가 단순한 2D 시각 정보에 머무르지 않고,

- 장면을 3D로 이해할 수 있는 데이터
- 고품질 렌더링에 활용할 수 있는 데이터
- 프로젝트 내부에서 설득력 있게 시각화할 수 있는 데이터
- 외부 3D 툴로 전달 가능한 데이터

로 변환되도록 하는 것이 목표이다.

---

## 2. 문제 정의

기존의 컴퓨터 비전 기술은 이미지 속 객체를 인식하거나, 객체의 영역을 분리하거나, 단일 이미지로부터 깊이를 추정하는 데까지는 상당히 발전해 있다. 예를 들어 객체 탐지 기술은 이미지나 영상에서 객체의 위치와 종류를 찾는 작업이며, [Ultralytics YOLO 문서](https://docs.ultralytics.com/tasks/detect/)에서도 출력이 주로 **bounding box, class label, confidence score** 중심임을 설명한다. 이는 “무엇이 어디 있는가”를 빠르게 파악하는 데에는 매우 유용하지만, 기본적으로는 **2D 이미지 평면 위 정보**에 머무른다.

객체 분할 기술도 매우 강력해졌다. [Segment Anything 공식 소개](https://ai.meta.com/research/publications/segment-anything/)에 따르면, Segment Anything은 범용 image segmentation을 위한 새로운 task, model, dataset을 제안하며 다양한 객체 mask 생성에 활용될 수 있다. 이러한 기술은 객체의 경계를 정밀하게 구분하고 객체별 영역을 분리하는 데 유용하지만, segmentation 결과만으로는 객체의 **실제 3차원 위치, 크기, 방향, 장면 내 공간적 관계**까지 직접 얻을 수는 없다. 즉, **객체의 경계는 알 수 있어도 장면의 3D 구조는 별도의 추론과 구조화가 필요하다.**

단일 이미지 기반 깊이 추정 역시 중요한 기반 기술이다. 최근에는 [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)와 [Apple Depth Pro GitHub](https://github.com/apple/ml-depth-pro)처럼 더 최신의 monocular depth 계열이 등장하여, 단일 이미지에서 보다 강력한 depth 추정을 제공하고 있다. 그러나 이러한 결과도 일반적으로는 **픽셀 단위 depth map** 또는 **상대적/추정된 절대 깊이 정보**에 가깝고, 이를 곧바로 **객체 단위의 3D 장면 표현**이나 **그래픽 렌더링용 구조화 데이터**로 연결하려면 추가적인 해석과 통합 과정이 필요하다.

최근에는 장면의 여러 3D 속성을 함께 추론하려는 시도도 등장하고 있다. [VGGT 프로젝트 페이지](https://vgg-t.github.io/)는 하나의 feed-forward 모델이 **camera parameters, point maps, depth maps, 3D point tracks** 같은 핵심 3D 속성을 직접 추론하는 방향을 제시하고, [DUSt3R 공식 소개](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)는 pointmap 기반으로 camera calibration이나 viewpoint pose에 대한 강한 사전정보 없이 3D reconstruction을 다루는 방향을 보여준다. 또한 [Wonderland 프로젝트 페이지](https://snap-research.github.io/wonderland/)는 단일 이미지에서 **넓은 범위의 3D scene**을 생성하려는 방향을 제시한다. 이러한 흐름은 기존보다 훨씬 “장면 전체의 3D 이해”에 가까운 접근이다.

하지만 본 프로젝트가 목표로 하는 것은 여기서 한 단계 더 나아간다. 이 프로젝트는 단순히 3D geometry를 추정하는 것만이 아니라, **고수준 그래픽 표현을 위해 필요한 표면/재질/시각 속성까지 함께 다루고**, 그 결과를 **내 프로젝트 안에서 보기 좋게 시각화**하며, 동시에 **Unity, Blender, Unreal 같은 외부 3D 툴에서도 다시 활용 가능한 형태로 저장**하는 것을 목표로 한다. 이 방향은 [MAGE (Material-Aware 3D Geometry Estimation)](https://cvpr.thecvf.com/virtual/2025/poster/35228), [SF3D](https://cvpr.thecvf.com/virtual/2025/poster/32882), [3DTopia-XL 프로젝트 페이지](https://3dtopia.github.io/3DTopia-XL/), [Material Palette 프로젝트 페이지](https://astra-vision.github.io/MaterialPalette/), [DualMat 논문](https://arxiv.org/html/2508.05060v1), [Material Anything GitHub](https://github.com/3DTopia/MaterialAnything) 같은 최신 흐름과도 닿아 있다. 또한 [Khronos의 glTF 소개 페이지](https://www.khronos.org/gltf/)와 [Blender glTF 2.0 매뉴얼](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html), [Unity glTFast 문서](https://docs.unity3d.com/Packages/com.unity.cloud.gltfast%406.14/manual/ImportEditor.html), [Unreal glTF import 문서](https://dev.epicgames.com/documentation/en-us/unreal-engine/importing-gltf-files-into-unreal-engine)는 실제 제작 파이프라인으로의 연결 가능성을 보여준다.

따라서 본 프로젝트의 핵심 문제는 기존의 탐지, 분할, depth 추정 기술을 단순히 사용하는 것이 아니라, 그 결과를 객체 단위로 통합하여 **각 객체의 의미 정보, 공간 정보, 형태 정보, 시각·재질 정보를 함께 추정하고**, 이를 **장면 전체의 3D 월드 정보와 렌더링 가능한 데이터**로 변환하는 **중간 표현과 시각화/저장 파이프라인**을 구축하는 데 있다. 이 점에서 본 프로젝트는 단순 객체 탐지 프로젝트와도 다르고, 단순 depth estimation 프로젝트와도 다르며, **2D 이미지 이해 결과를 3D 장면 구조와 그래픽 표현 데이터로 조직하는 시스템**에 가깝다.

---

## 3. 최종 목표

본 프로젝트의 최종 목표는 다음과 같다.

### 입력
- 단일 RGB 이미지 1장

### 처리
- 이미지 내의 구별 가능한 객체 탐지
- 객체별 영역 분리 또는 식별
- 깊이 정보 추정
- 객체별 3차원 위치 추정
- 객체별 형태 또는 대략적 3D 표현 생성
- 객체별 시각·재질 관련 정보 추정
- 장면 전체를 구조화된 scene data로 변환
- 프로젝트 내부 viewer에서 결과를 시각적으로 설득력 있게 표현
- 외부 3D 툴에서 활용할 수 있는 형식으로 export

### 출력
- 객체별 의미 정보와 공간 정보가 포함된 구조화된 장면 데이터
- 객체별 형태 정보와 시각·재질 정보가 포함된 렌더링 지향 데이터
- 프로젝트 내부 3D viewer에서 시연 가능한 결과
- JSON / glTF / GLB 등 외부 툴 연동을 고려한 export 결과

즉 최종 결과는 단순히 bbox가 그려진 이미지가 아니라,

- 장면 안에 어떤 객체들이 존재하는지
- 각 객체가 3D 공간에서 어디에 있는지
- 각 객체를 어떤 형태로 표현할지
- 고수준 그래픽 표현을 위해 어떤 표면/재질/텍스처 관련 정보를 가질지
- 이를 내부 시스템과 외부 툴에서 어떻게 재사용할 수 있을지

가 드러나는 결과물이어야 한다.

---

## 4. 현재 목표와 현실적 범위

본 프로젝트의 이상적인 최종 목표는 상당히 크기 때문에, 초기 단계부터 모든 기능을 한 번에 완성하는 것은 비현실적이다. 따라서 현재 단계에서는 **약식 프로토타입을 우선 구현**하는 것을 목표로 한다.

### 현재 단계에서의 현실적 목표
초기 프로토타입에서는 다음 수준을 달성하는 것을 목표로 한다.

1. 입력 이미지 1장을 받아 처리할 수 있다.  
2. 이미지 속 주요 객체들을 탐지할 수 있다.  
3. 객체별 대략적인 depth 정보를 얻을 수 있다.  
4. 각 객체의 중심 위치를 기준으로 대략적인 3D 위치를 계산할 수 있다.  
5. 각 객체를 단순 3D box 또는 primitive로 표현할 수 있다.  
6. 색상, 표면 속성, 텍스처 후보 정보 등 **그래픽 품질 향상에 활용 가능한 데이터의 초기 형태**를 정의하거나 일부 추정할 수 있다.  
7. 이를 이용해 기본적인 3D 장면 시각화를 수행할 수 있다.  
8. 결과를 이후 Unity / Blender / Unreal 연동이 가능하도록 저장 가능한 구조로 정리할 수 있다.

### 초기 단계에서 우선하지 않는 것
초기 프로토타입에서는 다음 항목들을 후순위로 둔다.

- 고정밀 카메라 보정
- 정교한 full mesh reconstruction
- 완벽한 scale estimation
- 완성형 PBR material reconstruction
- 완전한 texture restoration
- 복잡한 가림 처리
- 물리적으로 정확한 장면 재구성

중요한 점은, **그래픽스 정보가 덜 중요해서 미루는 것이 아니라**, 구현 난이도 때문에 **단계적으로 접근한다**는 것이다. 즉 공간 구조 추정과 그래픽 표현 데이터 추정은 둘 다 핵심 목표이며, 초기에는 단순 형태와 단순 시각 속성부터 시작해 점진적으로 고도화한다.

---

## 5. 시스템 동작 흐름

본 프로젝트의 전체 동작 흐름은 아래와 같다.

### 1) 입력 이미지 로드
사용자가 입력한 단일 이미지를 읽어온다.

### 2) 객체 탐지 및 객체 구분
이미지 내에 존재하는 구별 가능한 객체들을 탐지한다. 이 단계에서는 객체의 종류, 신뢰도, 2D bounding box, 필요하다면 segmentation mask 등을 얻는다. 객체 탐지와 분할에는 [Ultralytics YOLO 문서](https://docs.ultralytics.com/tasks/detect/)와 [Segment Anything 공식 소개](https://ai.meta.com/research/publications/segment-anything/) 같은 기반 기술을 참고할 수 있다.

### 3) 깊이 추정
입력 이미지 전체를 기준으로 depth map을 추정한다. 이를 통해 각 픽셀 또는 객체 영역이 카메라로부터 얼마나 떨어져 있는지에 대한 정보를 얻는다. 이 단계는 [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)나 [Apple Depth Pro GitHub](https://github.com/apple/ml-depth-pro) 같은 최신 monocular depth 계열을 참고할 수 있다.

### 4) 객체별 깊이 분석
각 객체에 대해 중심점 depth, 평균 depth, 또는 대표 depth 값을 계산한다. 이를 통해 각 객체의 대략적인 3차원 위치 계산에 필요한 정보를 만든다.

### 5) 2D 정보와 depth를 이용한 3D 위치 추정
객체의 2D 위치와 depth 정보를 결합하여, 각 객체를 카메라 좌표계 또는 장면 좌표계 상의 3D 위치로 변환한다.

### 6) 객체의 단순 3D 표현 생성
초기에는 각 객체를 box, plane, point, billboard, coarse mesh 등의 단순 표현으로 다룬다. 이 단계에서 객체의 대략적인 크기와 배치를 함께 결정한다.

### 7) 시각·재질 관련 정보 추정
객체 또는 장면으로부터 렌더링 품질 향상에 쓸 수 있는 정보를 추정한다. 예를 들어 다음이 포함될 수 있다.

- 대표 색상
- texture 후보
- albedo 성격의 색 정보
- normal / roughness / metallic로 확장 가능한 표면 속성
- 조명과 재질을 분리하기 위한 보조 정보

초기에는 완전한 PBR material reconstruction이 아니어도 되며, **고품질 렌더링에 활용 가능한 속성 구조를 정의하고 일부 채워 넣는 것**이 중요하다.

### 8) scene representation 생성
각 객체의 정보를 모아 장면 전체를 표현하는 구조화된 scene data를 생성한다. 이 데이터는 이후 저장, 재사용, 시각화, 확장에 사용된다.

### 9) 내부 3D 시각화
생성된 scene data를 바탕으로 장면을 3D 공간에 배치하여 시각화한다. 이 viewer는 단순 디버그 도구가 아니라, 프로젝트의 최종 결과를 설득력 있게 보여주는 **주요 산출물** 중 하나다.

### 10) 외부 툴 연동용 export
생성된 장면 정보를 JSON, glTF, GLB 등 구조화된 형식으로 저장하여, 향후 Unity, Blender, Unreal 같은 외부 3D 툴에서도 활용 가능하도록 정리한다. [Khronos의 glTF 소개 페이지](https://www.khronos.org/gltf/)와 [Blender glTF 2.0 매뉴얼](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html), [Unity glTFast 문서](https://docs.unity3d.com/Packages/com.unity.cloud.gltfast%406.14/manual/ImportEditor.html), [Unreal glTF import 문서](https://dev.epicgames.com/documentation/en-us/unreal-engine/importing-gltf-files-into-unreal-engine)는 이 방향의 현실적인 기준이 된다.

---

## 6. 핵심 기술 요소

### 1) 객체 탐지 및 분할
입력 이미지에서 객체를 찾고, 가능하면 객체별 영역을 분리해야 한다. [Ultralytics YOLO 문서](https://docs.ultralytics.com/tasks/detect/)는 detection 중심의 대표적 레퍼런스이고, [Segment Anything 공식 소개](https://ai.meta.com/research/publications/segment-anything/)는 범용 segmentation의 강력한 출발점이다.

### 2) 단일 이미지 기반 깊이 추정
입력 이미지가 한 장뿐이므로 stereo 정보 없이 monocular depth estimation 기법을 사용해야 한다. [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)와 [Apple Depth Pro GitHub](https://github.com/apple/ml-depth-pro)는 이 단계의 최신 참고 대상이다.

### 3) 2D에서 3D로의 변환
객체의 이미지 좌표와 depth 정보를 조합하여, 이를 3D 좌표계상의 위치로 바꾸는 과정이 필요하다. 이 부분은 카메라 모델, 좌표계 정의, 투영 관계 등에 대한 이해가 필요하다.

### 4) 객체 형태 표현
초기에는 box와 같은 primitive 형태로 시작하더라도, 장기적으로는 coarse mesh, textured mesh, category-based shape 등으로 확장 가능한 구조가 필요하다. [Wonder3D 프로젝트 페이지](https://www.xxlong.site/Wonder3D/), [SF3D](https://cvpr.thecvf.com/virtual/2025/poster/32882), [3DTopia-XL 프로젝트 페이지](https://3dtopia.github.io/3DTopia-XL/)는 이 축의 좋은 참고 레퍼런스다.

### 5) 그래픽 표현용 속성 추정
이 프로젝트는 비전만이 아니라 그래픽스도 동등한 우선순위로 다룬다. 따라서 geometry만이 아니라 **표면과 재질을 더 잘 표현하기 위한 데이터**도 중요하다. 이 데이터는 texture map, albedo, normal, roughness, metallic, reflectance 성격의 값일 수 있으며, [MAGE](https://cvpr.thecvf.com/virtual/2025/poster/35228), [Material Palette 프로젝트 페이지](https://astra-vision.github.io/MaterialPalette/), [DualMat 논문](https://arxiv.org/html/2508.05060v1), [Material Anything GitHub](https://github.com/3DTopia/MaterialAnything) 같은 최신 레퍼런스가 이 방향을 잘 보여준다.

### 6) Scene Representation 설계
이 프로젝트의 핵심은 특정 모델 하나가 아니라, 다양한 추론 결과를 어떤 구조로 저장하고 연결할 것인지에 있다. [VGGT 프로젝트 페이지](https://vgg-t.github.io/)나 [DUSt3R 공식 소개](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/) 같은 최근 연구는 장면의 여러 3D 속성을 통합적으로 추론하는 흐름을 보여주지만, 본 프로젝트는 이를 **객체 단위 scene data + 렌더링 속성 + viewer + export**까지 연결해야 한다.

### 7) 내부 시각화 시스템
결과를 단순 수치나 파일로만 저장하는 것이 아니라, **내 프로젝트 안에서 멋있게 띄우는 것 자체가 목표**이므로, viewer와 렌더링 표현도 핵심 기술 요소로 봐야 한다.

### 8) 외부 툴 호환 저장 형식
[Khronos의 glTF 소개 페이지](https://www.khronos.org/gltf/)는 glTF를 3D scenes and models의 효율적 전달과 상호운용성을 위한 표준 형식으로 설명하고, [Blender glTF 2.0 매뉴얼](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html), [Unity glTFast 문서](https://docs.unity3d.com/Packages/com.unity.cloud.gltfast%406.14/manual/index.html), [Unreal glTF import 문서](https://dev.epicgames.com/documentation/en-us/unreal-engine/importing-gltf-files-into-unreal-engine)는 실제 툴 연동 관점에서 참고할 수 있다. 따라서 glTF / GLB 중심 export는 본 프로젝트의 현실적인 외부 연동 목표가 될 수 있다.

---

## 7. 프로젝트 결과물 형태

본 프로젝트의 결과물은 크게 세 가지 관점에서 볼 수 있다.

### 1) 내부 시각화 결과물
- 입력 이미지
- 탐지된 객체 결과
- depth 결과
- 3D 공간에 배치된 객체들
- 객체의 표면 / 재질 정보를 반영한 시각화
- 장면을 회전·확인할 수 있는 viewer

즉 시연할 때는 “이미지를 넣으면 장면 안 객체들이 3D 공간에 이렇게 배치되고, 그래픽적으로도 이런 식으로 표현된다”는 흐름을 보여줄 수 있어야 한다.

### 2) 내부 데이터 결과물
내부적으로는 각 객체가 다음과 같은 정보 단위로 저장되어야 한다.

- 객체 종류
- 신뢰도
- 2D bounding box / mask
- 객체 중심점
- depth 정보
- 3D 위치 / 회전 / 크기
- primitive 또는 mesh 정보
- texture / material / surface property 정보
- 추정 신뢰도 및 메타데이터

즉 이 프로젝트의 핵심 산출물은 단순 시각화 화면뿐 아니라, **장면을 구조화하여 저장한 데이터 자체**이기도 하다.

### 3) 외부 연동 결과물
- JSON 기반 scene description
- glTF / GLB 기반 3D scene 또는 asset export
- Unity / Blender / Unreal에서 다시 읽을 수 있는 데이터 구조

교수님 관점에서는 이 부분이 특히 중요하다. 이는 프로젝트가 단순 데모를 넘어서 **재사용 가능하고 확장 가능한 시스템**이라는 점을 보여주기 때문이다. [Khronos의 glTF 소개 페이지](https://www.khronos.org/gltf/)는 glTF가 이런 상호운용성 목적에 맞는 형식임을 보여준다.

---

## 8. 이번 주 목표

현재 시점에서 이번 주까지의 목표는 최종 완성이 아니라, 다음 미팅 전까지 **관련 기술을 조사하고, 단순한 이미지 기준으로라도 돌아가는 약식 프로토타입을 만드는 것**이다.

### 이번 주 핵심 목표
1. 프로젝트의 전체 입력-출력 구조를 정리한다.  
2. “비전 결과 + 그래픽 표현 정보 + export 가능 구조”를 포함하는 최소 scene schema를 정의한다.  
3. 단일 이미지 입력 기준으로 최소한의 end-to-end 흐름을 만든다.  
4. 객체 탐지, depth, 3D 배치 중 일부가 실제 구현 또는 더미 형태로라도 연결되도록 한다.  
5. 프로젝트 내부에서 결과를 띄울 수 있는 최소 viewer 흐름을 잡는다.  
6. 향후 Unity / Blender / Unreal export를 고려한 저장 구조 초안을 만든다.

### 이번 주 성공 기준
이번 주가 끝났을 때 최소한 다음 중 상당 부분이 가능해야 한다.

- 입력 이미지 하나를 받아 처리할 수 있다.
- 객체 탐지 결과를 확인할 수 있다.
- depth 또는 대략적인 거리 추정 결과를 확인할 수 있다.
- 객체 하나 이상을 3D 공간에 단순 형태로 배치할 수 있다.
- 시각·재질 관련 속성을 담을 데이터 구조가 정의되어 있다.
- 내부 viewer 또는 최소 시각화 출력이 있다.
- 외부 툴 연동을 위한 저장 포맷 방향이 설명 가능하다.

즉 이번 주 목표는 논문 수준의 정확한 재구성이 아니라, **아이디어가 아니라 실제 구현 흐름으로 시작되었고, 그 흐름이 그래픽스와 export까지 고려하고 있다는 증거를 만드는 것**이다.

---

## 9. 향후 확장 방향

### 1) 객체 형태 정교화
단순 box 표현을 넘어, 카테고리 기반 3D shape, point cloud, coarse mesh, textured mesh 방향으로 확장할 수 있다. [Wonder3D 프로젝트 페이지](https://www.xxlong.site/Wonder3D/), [SF3D](https://cvpr.thecvf.com/virtual/2025/poster/32882), [3DTopia-XL 프로젝트 페이지](https://3dtopia.github.io/3DTopia-XL/)은 single-image to 3D asset 흐름의 좋은 참고 레퍼런스다.

### 2) 텍스처와 재질 정보 강화
대표 색상 수준을 넘어 texture map, albedo, normal, roughness, metallic 등 PBR 지향 속성으로 확장할 수 있다. [Material Palette 프로젝트 페이지](https://astra-vision.github.io/MaterialPalette/), [DualMat 논문](https://arxiv.org/html/2508.05060v1), [Material Anything GitHub](https://github.com/3DTopia/MaterialAnything)는 이 방향의 직접적인 참고 대상이다.

### 3) 카메라 모델 개선
현재는 단순 intrinsics 가정으로 시작하더라도, 이후 카메라 보정이나 투영 모델을 개선할 수 있다. [VGGT 프로젝트 페이지](https://vgg-t.github.io/)와 [DUSt3R 공식 소개](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)는 geometry와 camera parameters를 더 직접적으로 다루는 참고 사례다.

### 4) Scene Quality 향상
객체 간 가림 관계, 물체의 접지 여부, 크기 일관성, 상대적 배치 등을 더 정교하게 다룰 수 있다. [Wonderland 프로젝트 페이지](https://snap-research.github.io/wonderland/)는 단일 이미지에서 더 넓은 범위의 3D scene 생성 방향을 보여준다.

### 5) 외부 툴 연동 강화
초기에는 JSON + glTF / GLB 수준으로 시작하더라도, 이후 Blender round-trip, Unity prefab화, Unreal level import 등 파이프라인 연동으로 발전시킬 수 있다. [Blender glTF 2.0 매뉴얼](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html), [Unity glTFast 문서](https://docs.unity3d.com/Packages/com.unity.cloud.gltfast%406.14/manual/ImportEditor.html), [Unreal glTF import 문서](https://dev.epicgames.com/documentation/en-us/unreal-engine/importing-gltf-files-into-unreal-engine)는 이 방향의 현실성을 보여준다.

### 6) 연구 및 졸업작품 수준 확장
이 프로젝트는 “단일 이미지 기반 scene understanding”, “rendering-oriented attribute estimation”, “tool-compatible scene export”를 하나로 통합한다는 점에서 졸업 프로젝트로서 구현성과 확장성을 함께 보여줄 수 있다.

---

## 10. 참고 레퍼런스 및 유사 사례

본 프로젝트와 완전히 동일한 상용 서비스가 있는 것은 아니지만, 유사한 방향의 연구와 서비스는 크게 다섯 부류로 정리할 수 있다.

### (1) 기반 기술
- [Ultralytics YOLO 문서](https://docs.ultralytics.com/tasks/detect/) : 객체 탐지
- [Segment Anything 공식 소개](https://ai.meta.com/research/publications/segment-anything/) : 범용 segmentation

### (2) 깊이 추정
- [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- [Apple Depth Pro GitHub](https://github.com/apple/ml-depth-pro)

### (3) 장면 단위 3D 이해
- [VGGT 프로젝트 페이지](https://vgg-t.github.io/) : camera parameters, point maps, depth maps, 3D point tracks를 직접 추론
- [DUSt3R 공식 소개](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/) : pointmap 기반 3D reconstruction
- [Wonderland 프로젝트 페이지](https://snap-research.github.io/wonderland/) : single image에서 wide-scope 3D scene 생성

### (4) 단일 이미지 → 3D 오브젝트 / 그래픽 품질
- [Wonder3D 프로젝트 페이지](https://www.xxlong.site/Wonder3D/)
- [SF3D](https://cvpr.thecvf.com/virtual/2025/poster/32882)
- [MAGE](https://cvpr.thecvf.com/virtual/2025/poster/35228)
- [3DTopia-XL 프로젝트 페이지](https://3dtopia.github.io/3DTopia-XL/)

### (5) 재질 / PBR / material-aware generation
- [Material Palette 프로젝트 페이지](https://astra-vision.github.io/MaterialPalette/)
- [DualMat 논문](https://arxiv.org/html/2508.05060v1)
- [Material Anything GitHub](https://github.com/3DTopia/MaterialAnything)

### (6) 외부 툴 / 파이프라인 연동 기준
- [Khronos의 glTF 소개 페이지](https://www.khronos.org/gltf/)
- [Blender glTF 2.0 매뉴얼](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html)
- [Unity glTFast 문서](https://docs.unity3d.com/Packages/com.unity.cloud.gltfast%406.14/manual/ImportEditor.html)
- [Unreal glTF import 문서](https://dev.epicgames.com/documentation/en-us/unreal-engine/importing-gltf-files-into-unreal-engine)

### (7) 산업 / 서비스 사례
- [Tripo 공식 사이트](https://www.tripo3d.ai/)
- [Meshy 공식 사이트](https://www.meshy.ai/)
- [Kaedim 공식 사이트](https://kaedim3d.com/)

따라서 본 프로젝트는 기존의 detection, segmentation, depth estimation, scene geometry 추론, material-aware 3D generation, 3D interchange 포맷 흐름 사이에서, **객체 단위 scene structuring + rendering-oriented data estimation + internal visualization + external tool export**를 직접 구현하는 졸업 프로젝트형 시스템으로 위치시킬 수 있다.

---

## 11. 정리

Image-to-World 프로젝트는 단일 이미지 속 장면을 단순히 인식하는 데서 끝나지 않고, 이를 **객체 단위의 의미 정보, 공간 정보, 형태 정보, 시각·재질 정보를 포함한 3차원 장면 데이터**로 변환하는 것을 목표로 한다.

현재 단계에서는 완성형 reconstruction이나 완전한 PBR 추정보다, **객체 탐지 → 깊이 추정 → 3D 위치 계산 → 단순 3D 표현 → 시각 속성 구조화 → 내부 시각화 → 외부 툴 export**의 전체 흐름이 작동하는 프로토타입을 우선 구축한다.

이 프로젝트는 이후 shape, texture, material, camera model, scene quality, export pipeline 등 다양한 방향으로 확장 가능하며, 졸업 프로젝트로서도 **구현성, 시연성, 확장성, 활용성**을 함께 보여줄 수 있는 주제이다.