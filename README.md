# Image-to-World

## 소개

Image-to-World는 단일 이미지를 바탕으로 실내 장면을 3D로 복원하는 방향의 프로젝트입니다.

처음에는 객체 하나하나를 3D로 만드는 쪽도 생각했지만,  
현재는 **장면 전체를 3D 씬으로 복원하고, 나중에는 3D 툴에서 활용 가능한 형태로 만드는 것**을 목표로 잡고 있습니다.

아직 세부 구현은 확정하지 않았고, 관련 논문과 레퍼런스를 보면서 방향을 구체화하는 단계입니다.

---

## 현재 생각 중인 방향

현재는 다음과 같은 흐름을 생각하고 있습니다.

- 이미지에서 장면 구성 요소 분리
- 깊이 / 구조 / 배치 추정
- 객체별 3D 복원 또는 생성
- 하나의 3D scene 형태로 조립
- 이후 활용 가능한 형태로 정리

세부적인 구현 방식은 아직 조사 중입니다.

---

## 현재 우선순위

지금은 기능을 많이 늘리기보다, 기본 성능을 먼저 보는 쪽으로 생각하고 있습니다.

예를 들면:

- 객체 분리가 안정적인지
- 장면 배치가 자연스러운지
- 가려진 부분 복원이 어느 정도 되는지
- 결과를 scene 형태로 정리할 수 있는지

---

## 주요 레퍼런스

현재 우선적으로 참고하고 있는 레퍼런스는 아래와 같습니다.

- Zero-Shot Scene Reconstruction from Single Images with Deep Prior Assembly
- Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling
- 3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework
- DepR: Depth Guided Single-view Scene Reconstruction with Instance-level Diffusion
- InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes
- PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction
- Gen3DSR: Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View

---

## 현재 상태

- 프로젝트 방향 정리 중
- 관련 논문 조사 중
- 전체 파이프라인 구상 중
- README는 임시 버전이며, 이후 수정 예정

---

## 앞으로

논문들을 조금 더 읽어본 뒤,  
프로젝트 목표 / 파이프라인 / 구현 계획을 다시 정리할 예정입니다.