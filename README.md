# Image-to-World

단일 이미지를 입력받아 장면 내 객체를 분리하고, 깊이와 객체별 3D 복원을 활용해 최종적으로 **3D 월드 기반 장면 정보**로 재구성하는 프로젝트입니다. 본 프로젝트는 하나의 거대한 end-to-end 생성 모델을 직접 학습하는 방식보다, **모듈형 파이프라인**을 구성해 각 단계를 독립적으로 검증하고 점진적으로 고도화하는 방향을 채택합니다. 

## 1. 프로젝트 개요

이 프로젝트의 큰 목표는 **단일 이미지 한 장으로부터 장면에 존재하는 객체들의 위치와 형태를 추정하고, 이를 3차원 장면 정보로 변환하는 것**입니다. 다만 현재 단계에서는 PixARMesh 같은 통합 생성 모델을 직접 학습하는 방식보다, **객체 분리 → 깊이 추정 → 객체 보완 → 객체별 3D 생성 → 장면 배치 → scene assembly**로 이어지는 파이프라인을 먼저 구축하는 것이 더 현실적이라고 판단했습니다. Deep Prior Assembly는 zero-shot 조립형 구조를, Gen3DSR는 divide-and-conquer 기반 모듈형 구조를, 3D-RE-GEN은 editable scene을 위한 compositional generation 구조를 제시합니다.

본 프로젝트는 최종적으로 다음과 같은 결과를 목표로 합니다.

- 입력 이미지 내 객체를 인스턴스 단위로 분리
- 장면의 깊이와 기하 정보를 추정
- 가려진 객체를 가능한 범위에서 보완
- 객체별 3D 자산 생성
- 객체를 장면 좌표계에 배치
- 하나의 3D scene으로 조립
- 추후 obj + mtl / fbx / glTF 등 활용 가능한 형태로 출력

---

## 2. 파이프라인

### 01_tag

입력 이미지에서 장면에 존재하는 객체 후보 태그를 추출합니다.

### 02_mask

태그를 바탕으로 객체 인스턴스를 검출하고, 객체별 마스크 및 crop 이미지를 생성합니다.

### 03_inpaint

부분적으로만 보이는 객체를 더 완전한 형태로 보완하기 위해 inpainting을 수행합니다.

### 04_asset

보완된 객체 이미지를 바탕으로 객체별 3D 어셋(mesh)을 생성합니다.

### 05_depth

원본 이미지의 depth map을 추정하여 장면 내 상대적 거리 정보를 얻습니다.

### 06_transform

객체의 2D 위치, depth, 3D 어셋 정보를 이용해 장면 내 배치 좌표를 계산합니다.

### 07_assemble

각 객체 어셋을 최종 장면 좌표계에 배치하여 하나의 scene mesh로 합칩니다.

---

## 3. 현재까지 수행한 작업 (DONE)

현재는 **최소 end-to-end 프로토타입이 실제로 연결된 상태**입니다.

### 전체 파이프라인 구성

- 태그 추출부터 scene assembly까지 전체 흐름이 실제 코드로 연결되어 있음
- 각 단계별 중간 결과물을 저장하고 확인할 수 있는 구조를 갖춤
- 객체별 결과를 기반으로 후속 단계 입력을 생성하는 형태로 구성됨
- 사용 방식:
  - 모듈형 파이프라인 구조
  - 단계별 중간 산출물 저장 기반 디버깅
  - 후속 단계가 이전 단계 결과를 입력으로 받는 순차 처리 방식

### 01_tag

- 입력 이미지에서 태그를 추출하는 기능 구현
- 사용 기술:
  - **RAM++ (Recognize Anything Model Plus)** 기반 태그 추출
- 객체가 아닌 배경성 태그가 너무 많이 포함되는 문제를 완화하기 위해 **임시 필터링** 적용
- 현재는 `room`, `floor`, `wall` 등 배경성 태그를 일부 제외하는 식의 임시 대응을 사용 중
- 현재 이슈:
  - 객체가 아닌 사물도 태그로 추가되는 문제
  - 간혹 객체를 잘 인식하지 못하는 문제
  - 우선 배경 분리 이후 다시 검토 필요

### 02_mask

- 태그 기반 객체 마스크 추출 파이프라인 구현
- 사용 기술:
  - **Grounding DINO** 기반 텍스트 조건 객체 검출
  - **SAM2 (Segment Anything Model 2)** 기반 객체 마스크 생성
- 객체별 mask, crop, RGBA crop 등을 저장하는 구조 확보
- 전반적인 결과는 나쁘지 않은 수준
- 현재 이슈:
  - 상위 단계인 tag 품질에 영향을 받기 때문에 명확한 평가는 아직 어려움

### 03_inpaint

- 객체 crop을 확장하여 보완된 이미지를 생성하는 기능 구현
- 사용 기술:
  - **Stable Diffusion XL Inpainting**
- 결과물 자체는 나쁘지 않은 수준
- 현재 이슈:
  - 속도가 매우 느려 전체 파이프라인 병목 중 하나로 판단됨

### 04_asset

- 객체 이미지로부터 3D 어셋을 생성하는 기능 구현
- 사용 기술:
  - **Shap-E** 기반 2D 이미지 → 3D mesh 생성
- 생성된 객체의 전체적인 형상은 대체로 나쁘지 않은 편
- 축 보정 관련 실험도 일부 반영됨
- 현재 이슈:
  - mesh 품질이 낮고 노이즈가 많음
  - 폴리곤 수가 지나치게 많음
  - 텍스처까지 입힐 수 있는 더 나은 3D 어셋 생성 모델이 필요함

### 05_depth

- depth map 추정 기능 구현
- 사용 기술:
  - **Depth Anything V2**
- 결과 이미지 저장 가능
- 현재는 정량 평가 없이 육안 확인 위주로 검토 중
- 현재 이슈:
  - 깊이 추정 결과 이미지를 눈으로 확인할 수밖에 없어 평가가 모호함
  - 다만 현재로서는 큰 문제는 없어 보임

### 06_transform

- depth와 객체 정보를 활용해 장면 내 배치 좌표를 계산하는 기능 구현
- 사용 기술:
  - 객체 bounding box / mask 기반 2D 위치 정보 활용
  - depth map 기반 상대 거리 추정
  - pseudo camera / pinhole 가정 기반의 초기 3D 배치 계산
  - 객체 크기와 depth를 결합한 heuristic scale 추정
- 객체들이 대략적인 위치에 놓이는 수준까지는 도달함
- transform 결과를 기준으로 얼추 맞는 배치가 나오는지 확인 가능한 상태
- 현재 이슈:
  - 완성도를 높이려면 많은 수정이 필요함
  - 위치는 얼추 맞지만 명확하지 않음
  - 좌표계 일관성, 위치 정확도, 배치 로직 정교화가 필요함

### 07_assemble

- transform 결과와 객체 어셋을 이용해 하나의 scene mesh로 합치는 기능 구현
- 사용 기술:
  - OBJ 단위 mesh 로드 및 병합
  - 객체별 transform 적용
  - scene 단위 mesh export
  - 객체별 색상 구분을 통한 시각화용 출력
- 최종 장면 형태로 결과를 묶어 확인할 수 있는 단계까지 연결됨
- scene 단위 결과물을 출력할 수 있는 최소 구조는 확보됨
- 현재 이슈:
  - 최종 배치 결과의 완성도가 아직 충분히 높지 않음
  - transform 단계와의 책임 분리가 더 명확해질 필요가 있음
  - 축 보정, 배치 반영 방식, scene 출력 품질 개선이 필요함

### 시각화 및 디버깅 기반 확보

- 단계별 결과물을 직접 확인하면서 품질을 판단할 수 있는 구조가 마련됨
- 현재 프로젝트는 “품질 개선을 위한 기반”까지는 확보된 상태로 볼 수 있음
- 사용 방식:
  - 단계별 이미지 저장
  - 객체별 중간 산출물 저장
  - 최종 scene 결과물 export 및 외부 툴 확인 가능 구조

![INPUT](./doc/260327/input.jpg)
![FRONT](./doc/260327/front.png)
![SIDE](./doc/260327/side.png)
![TOP](./doc/260327/top.png)

---

## 4. 앞으로 수행할 작업 (TODO)

앞으로의 작업은 크게 **정확도 개선**, **속도 개선**, **결과물 품질 개선**, **구조 정리**로 나눌 수 있습니다.

### A. 태그 및 마스크 품질 개선

- 객체 태그와 장면 태그를 분리 저장하는 구조로 개선
- 단순 필터링이 아니라 배경/구조물 처리까지 고려한 태그 체계 정리
- 배경 분리 이후 tag 단계 결과를 다시 검증
- tag 결과 품질이 개선된 뒤 mask 단계 성능 재평가

### B. inpainting 속도 및 품질 개선

- 현재 inpainting 병목 원인 분석
- 필요 시 더 빠른 설정 또는 대체 모델 검토
- 마스크 설계 방식 개선
- 객체 유형별 프롬프트 또는 생성 전략 개선
- 여러 실험을 빠르게 반복할 수 있도록 추론 시간 단축

### C. 3D 어셋 품질 개선

- mesh 노이즈 제거 후처리 실험
- 과도한 폴리곤 수를 줄이기 위한 decimation / remeshing 검토
- 더 높은 품질의 2D→3D 생성 모델 조사 및 교체 실험
- 최종적으로는 텍스처까지 포함 가능한 모델 도입 검토

### D. depth 검증 체계 보강

- 현재 depth 결과를 단순 이미지 확인에만 의존하지 않도록 개선
- 객체 위치 및 배치 결과와 함께 depth 품질을 간접 검증하는 방식 도입
- 필요 시 depth 통계값, 객체별 depth 샘플링 방식 개선

### E. transform / assemble 정교화

- 좌표계 정의를 명확히 문서화
- transform 단계와 assemble 단계의 책임 분리
- 객체 위치 계산 기준 개선
- scale 추정 방식 개선
- 축 반전 및 회전 보정 로직 정리
- 결과 장면을 더 자연스럽게 보이도록 배치 로직 개선

### F. 배경 / 구조물 복원 추가

- floor / wall plane 추정
- 배경 또는 구조물 mesh 생성 방향 검토
- 객체만 있는 scene이 아니라 실제 장면 구조를 갖춘 reconstruction으로 확장

### G. 코드 구조 및 유지보수성 개선

- 파일별 역할 재정리
- 하드코딩 최소화
- 설정 파일 분리
- 디버그/시각화/핵심 로직 분리
- 중간 산출물 관리 체계 정리

---

## 5. 테스트 이미지 및 참고 자료

### 테스트 이미지

- [pixabay simple room image](https://pixabay.com/ko/photos/%ec%9d%b8%ed%85%8c%eb%a6%ac%ec%96%b4-%eb%94%94%ec%9e%90%ec%9d%b8-%ed%98%84%eb%8c%80%ec%a0%81%ec%9d%b8-%ec%8a%a4%ed%83%80%ec%9d%bc-4467768/)

### 주요 참고 자료

- [Zero-Shot Scene Reconstruction from Single Images with Deep Prior Assembly](https://arxiv.org/html/2410.15971v1)
- [Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](https://arxiv.org/html/2411.19492v2)
- [3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework](https://arxiv.org/html/2512.17459v1)
- [DepR: Depth Guided Single-view Scene Reconstruction with Instance-level Diffusion](https://arxiv.org/html/2507.22825v1)
- [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](https://arxiv.org/html/2507.08416v2)
- [PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](https://arxiv.org/html/2603.05888v1)
- [Gen3DSR: Generalizable 3D Scene Reconstruction via Divide and Conquer from a Single View](https://arxiv.org/html/2404.03421v2)
- [Open-World Amodal Appearance Completion](https://arxiv.org/html/2411.13019v1)

