# 2026-05-29 교수님 시연 요약 (Image-to-World)

## 1) 시연 목표
- 단일 입력 이미지에서 객체 단위 3D 자산을 생성하고, 배경이 포함된 씬(GLB)으로 조립.
- `6126f0c4941a4db1dc591ba8fac79ea444ec1e9c` 시점 대비 **가시적 완성도 향상**을 확인.

## 2) 오늘 기준 핵심 결과
- 최종 씬 파일: `output/scene_assembly/raw_image0_assembled.glb`
- 최종 정합 비교: `output/eval/raw_image0/reproj_compare.png`
- 최신 평가 기록: `output/eval/manifest.jsonl` (tag: `M3-post-texture-final`, `M3-wall-up-018` 등)

## 3) 저번(6126f0c) 대비 개선 포인트
- 객체 가시성 개선:
  - 소스 카메라 관점에서 주요 객체(에어프라이어/병류) 동시 가시성 향상.
- 배경 복원 및 씬 구성:
  - LaMa 기반 clean background를 활용해 바닥/벽 평면 텍스처 씬 구성.
- 객체 주변 노이즈 완화:
  - crop 생성 시 마스크 정제(최대 연결요소 유지 + 경계 erosion) 적용.
- 평가 지표 개선:
  - 최근 실행 기준 `PSNR 13.3736`, `SSIM 0.7154`.

## 4) 교수님께 보여드릴 데모 순서 (권장)
1. 입력 이미지 확인
   - `data/raw_image0.jpg`
2. 최종 3D 씬 확인
   - `output/scene_assembly/raw_image0_assembled.glb`
3. 원본 vs 재투영 비교
   - `output/eval/raw_image0/reproj_compare.png`
4. 보조 데모(움직임/편집 가능성)
   - `output/demo/raw_image0/placed_motion.gif`
   - `output/demo/raw_image0/placed_grid.png`

## 5) 현재 한계 (솔직 보고)
- 일부 객체 메쉬 디테일(형상/표면)은 여전히 러프함.
- 카메라/좌표계 자동 최적화(WSL fitted transform)는 환경 이슈로 수동/규칙 보정이 일부 반영됨.
- 벽/바닥은 실제 기하 복원보다는 시연용 배경 평면 성격이 강함.

## 6) 다음 단계 계획
- 자동 피팅(WSL) 복구 후 수동 보정 제거.
- 객체별 메쉬 품질 개선(마스크 경계/입력 crop 품질 추가 개선).
- 발표 이후에는 정량 실험표(버전별 PSNR/SSIM + 시각 비교) 문서화.

## 7) 실행 기록(오늘 주요 검증)
- `python run.py --stage mask_postprocess --image-path data/raw_image0.jpg`
- `python run.py --stage crops_generation --image-path data/raw_image0.jpg`
- `python run.py --stage mesh_generation --image-path data/raw_image0.jpg`
- `python run.py --stage mesh_remesh --image-path data/raw_image0.jpg`
- `python run.py --stage mesh_texturing --image-path data/raw_image0.jpg`
- `python run.py --stage scene_assembly --image-path data/raw_image0.jpg`
- `python scripts/run_eval.py --image data/raw_image0.jpg --glb output/scene_assembly/raw_image0_assembled.glb --camera-json output/camera_estimation/raw_image0_perspective_fields.json --out-dir output/eval --max-width 1280 --tag M3-post-texture-final`

---
필요 시 본 문서 기반으로 PPT 1~2장 요약본으로 바로 변환 가능.
