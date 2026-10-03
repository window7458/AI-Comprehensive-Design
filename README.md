# 인공지능 종합설계 — 시각장애인 보행 보조를 위한 YOLO11s-seg 지식증류 파일럿

보도 위 장애물(기둥, 볼라드, 킥보드, 사람, 차량 …)을 **실시간 인스턴스 분할**하는 경량 모델(YOLO11s-seg)에
대형 비전 파운데이션 모델(VFM)의 표현을 **학습 때만** 증류해서, 추론 비용은 그대로 두고 성능을 올리는 것이 목표다.
이 저장소는 **어떤 teacher가 가장 좋은지 고르는 파일럿**을 한 번에 돌리는 코드다.
파일럿에서 고른 설정으로 전체 데이터 실험을 이어서 돌린다.

## 실험 구성

| ID | 트랙 | teacher / 인코더 | 추론 시 teacher 사용 |
|---|---|---|---|
| E0 | – | YOLO11s-seg baseline | ✗ |
| B1 | Distill | SigLIP2-B (`google/siglip2-base-patch16-512`) | ✗ |
| B2 | Distill | DINOv3-B (`facebook/dinov3-vitb16-pretrain-lvd1689m`) | ✗ |
| B2b *(선택)* | Distill | DINOv2-B (`facebook/dinov2-base`) | ✗ |
| B3 | Distill | DINOv3-B + SigLIP2-B 결합 teacher | ✗ |
| B4 | Distill | RADIOv2.5-B | ✗ |
| B5 | Distill | C-RADIOv3-B | ✗ |
| B5b *(선택)* | Distill | C-RADIOv4-SO400M (추론에 안 쓰니 teacher가 커도 됨) | ✗ |
| A-best | Fusion | B1–B5 중 val mask mAP가 가장 높은 **단일** 인코더 → 상한 비교용 | ✓ |

- **Distill (B\*)**: 학생 백본의 P3/P4/P5 (YOLO11s 4·6·10번 층) 출력을 1×1 conv adapter로 teacher 차원에 맞춘 뒤,
  teacher의 patch feature map과 **위치별 코사인 유사도 손실**(`1 - cos`)을 준다. 결합 teacher(B3)는 teacher마다 adapter를 따로 두고 손실을 평균낸다.
  학습이 끝나면 adapter를 떼어 낸 **일반 YOLO11s-seg 체크포인트(`student.pt`)** 를 저장하고, 평가도 이 파일로 한다. 배포 모델의 연산량은 E0와 같다.
- **Fusion (A-best)**: 고정된 teacher feature를 같은 층에 `BN(1×1 conv)`로 더한다. BN의 γ를 0으로 초기화해서 시작 상태가 baseline과 같다.
  추론 때도 teacher가 돌아가므로 배포용이 아니라 “teacher 정보를 직접 넣으면 어디까지 올라가나”를 보는 **상한선**이다.
- teacher는 모두 frozen이고 입력은 긴 변 512 px (ViT-B/16 기준 32×32 토큰)이다. 학습 이미지(모자이크 증강 포함)를 그대로 넣는다.

## 데이터: AI Hub 189 인도보행 Polygon, 26 → 10 클래스

### 원본 클래스별 분포 (26종)

| # | 클래스 | 인스턴스 | 비율 | 이미지 수 | 폴더 수 | → 학습 클래스 |
|---:|---|---:|---:|---:|---:|---|
| 1 | car | 147,132 | 23.69% | 55,957 | 1,800 | car |
| 2 | pole | 98,463 | 15.86% | 57,432 | 1,839 | obstacle |
| 3 | tree_trunk | 97,018 | 15.62% | 44,494 | 1,813 | obstacle |
| 4 | person | 47,192 | 7.60% | 26,149 | 1,691 | person |
| 5 | traffic_sign | 38,918 | 6.27% | 21,298 | 1,748 | *(제외)* |
| 6 | bollard | 37,266 | 6.00% | 13,906 | 1,605 | obstacle |
| 7 | truck | 33,211 | 5.35% | 20,190 | 1,647 | other_vehicle |
| 8 | traffic_light | 26,799 | 4.32% | 11,107 | 1,374 | traffic_light |
| 9 | movable_signage | 20,203 | 3.25% | 12,059 | 1,419 | obstacle |
| 10 | bus | 11,321 | 1.82% | 6,484 | 1,323 | bus |
| 11 | bicycle | 10,003 | 1.61% | 6,581 | 1,367 | bicycle |
| 12 | motorcycle | 9,039 | 1.46% | 6,731 | 1,261 | motorcycle |
| 13 | potted_plant\* | 9,031 | 1.45% | 4,300 | 1,074 | obstacle |
| 14 | bench | 6,215 | 1.00% | 3,225 | 756 | obstacle |
| 15 | power_controller | 5,262 | 0.85% | 3,446 | 876 | obstacle |
| 16 | barricade | 4,323 | 0.70% | 1,991 | 711 | obstacle |
| 17 | stop | 4,311 | 0.69% | 3,720 | 855 | obstacle |
| 18 | traffic_light_controller | 3,986 | 0.64% | 3,469 | 1,035 | obstacle |
| 19 | chair | 3,643 | 0.59% | 2,101 | 824 | obstacle |
| 20 | fire_hydrant | 2,593 | 0.42% | 2,290 | 864 | obstacle |
| 21 | carrier | 1,479 | 0.24% | 1,159 | 575 | other_vehicle |
| 22 | table | 1,317 | 0.21% | 860 | 400 | obstacle |
| 23 | kiosk | 1,198 | 0.19% | 947 | 448 | obstacle |
| 24 | stroller | 485 | 0.08% | 429 | 232 | other_vehicle |
| **25** | **scooter** | **351** | **0.06%** | **224** | **124** | **scooter** |
| 26 | wheelchair | 204 | 0.03% | 176 | 102 | other_vehicle |
| | 합계 | 620,963 | 100% | | | |

\* potted_plant는 원본 캡처가 잘려 근삿값.

### 10클래스 매핑 후 분포 (학습·평가 클래스)

| id | 학습 클래스 | 인스턴스 | 비율 | 합쳐진 원본 클래스 |
|---:|---|---:|---:|---|
| 0 | obstacle | 294,829 | 47.5% | 14종: pole, tree_trunk, bollard, movable_signage, potted_plant, bench, power_controller, barricade, stop, traffic_light_controller, chair, fire_hydrant, table, kiosk |
| 1 | car | 147,132 | 23.7% | car |
| 2 | person | 47,192 | 7.6% | person |
| 3 | other_vehicle | 35,379 | 5.7% | truck, carrier, stroller, wheelchair |
| 4 | traffic_light | 26,799 | 4.3% | traffic_light |
| 5 | bus | 11,321 | 1.8% | bus |
| 6 | bicycle | 10,003 | 1.6% | bicycle |
| 7 | motorcycle | 9,039 | 1.5% | motorcycle |
| 8 | **scooter** | **351** | **0.06%** | scooter |
| 9 | **stairs** | **0** | – | stairs (Polygon에는 없음 → Surface의 caution_zone/stairs를 `--extra-src`로 추가) |
| – | *(traffic_sign)* | *38,918* | *6.3%* | 미매핑 — 현재 제외 상태 |

- 매핑은 `configs/classes.yaml`의 `map` / `ignore`에 있고, `prepare_dataset.py`가 라벨을 **10클래스 id로** 쓴다.
  traffic_sign 폴리곤은 라벨에서 빠지므로 배경으로 학습된다. 원본에 있지만 표에 없는 라벨(cat, dog, parking_meter 등)도 버리고 개수만 기록한다.
- **stairs는 Polygon 데이터에 인스턴스가 0개**다. 클래스 자리(id 9)만 잡아 두었고 AP는 계산되지 않는다(요약표에서 빈칸).
  Surface 데이터의 caution_zone을 붙이기 전까지 파일럿 결과에는 영향이 없다.
- 요약표에 따로 나오는 평균: **key mAP** = obstacle, car, person, traffic_light, bus, bicycle, motorcycle (보행 안전 핵심),
  **rare mAP** = scooter, motorcycle, bicycle, bus (인스턴스 2% 미만). scooter AP는 별도 열로도 나온다.

원본 표 이미지: [26종 분포](docs/class_distribution_26.webp), [10클래스 매핑](docs/class_mapping_10.webp)

## 파인튜닝 포커스: COCO에 없는 클래스 + scooter / traffic_light

student는 COCO로 사전학습된 `yolo11s-seg.pt`에서 시작한다. car·person·bus·bicycle·motorcycle·truck은 COCO에 이미 있으니,
**COCO가 본 적 없는 클래스**와 보행 안전에 중요한 traffic_light를 중심으로 파인튜닝하도록 샘플링 가중치를 준다
(`configs/classes.yaml`의 `focus_weights`, 원본 클래스 기준).

| 가중치 | 원본 클래스 | 이유 |
|---:|---|---|
| 3.0 | scooter | COCO에 없음, 351개로 가장 희귀 |
| 2.0 | traffic_light | COCO에는 있지만 횡단 판단에 핵심 |
| 2.0 | wheelchair, stroller, carrier, kiosk | COCO에 없음, 희귀 |
| 1.5 | pole, bollard, tree_trunk, movable_signage, barricade, power_controller, traffic_light_controller, stop | COCO에 없음(이미 흔한 편) |
| 1.0 | car, person, bus, bicycle, motorcycle, truck, bench, chair, potted_plant, fire_hydrant, table | COCO에 있음 |

- 적용 위치: 클래스별 최소 이미지 수(`min-per-class × 가중치`), fill 단계 가중치, RFS 반복 횟수(`sqrt(t × 가중치 / f_c)`).
  모든 실험(E0, B1–B5, A-best)에 똑같이 적용되므로 비교는 공정하다.
- 9만 장짜리 가짜 인덱스로 6,000장 파일럿을 뽑아(아래 `include_all`은 끈 상태) 포커스를 끈 경우와 비교했다. traffic_light 이미지는 870 → 1,025장, other_vehicle은 1,934 → 2,199장으로 늘었다.
  scooter 이미지는 원래 전부(172장) 들어가 있어서 장수는 그대로지만, RFS 반복 후 학습 인스턴스가 568 → 1,014로 늘었다.
- 요약표에 **focus mAP**(scooter, traffic_light, stairs, obstacle, other_vehicle 평균)와 scooter·traffic_light 열이 따로 나온다.
- 끄려면 `python tools/make_subset.py ... --no-focus`.

### 해당 클래스가 든 사진은 전부 사용 (`include_all`)

`include_all: [scooter, stairs, traffic_light, wheelchair, stroller, carrier, kiosk]`에 있는 클래스가 **하나라도 든 사진은 전부** 쓴다.
사진 예산과 상관없이 먼저 넣고, 그 사진의 시퀀스가 속한 split(train/val/test)에 들어간다. 장수는 모두 **사진 수** 기준이다(한 사진에 신호등이 여러 개여도 1장).

| 클래스 | 오브젝트 | 사진 | 폴더 | 파일럿에 들어가는 사진 |
|---|---:|---:|---:|---|
| scooter | 351 | 224 | 124 | 224장 전부 |
| traffic_light | 26,799 | 11,107 | 1,374 | 11,107장 전부 (train 약 8.3천 장) |
| wheelchair / stroller / carrier / kiosk | 204 / 485 / 1,479 / 1,198 | 176 / 429 / 1,159 / 947 | | 전부 |
| stairs | 0 | 0 | 0 | Surface 데이터를 붙이면 전부 |

- traffic_light 사진만 1.1만 장이라 파일럿 train이 커진다. 기본 예산은 train 10,000장 / val 2,500장이고,
  필수 사진과 클래스별 최소 장수가 넘치면 예산을 넘겨서라도 넣는다.
  9만 장짜리 가짜 인덱스에서는 train 10,830장(RFS 반복 포함 학습 목록 15,782줄), val 2,500장이 나왔다.
- 학습 시간이 부담되면 `EXTRA="epochs=30" bash scripts/run_pilot.sh`처럼 epoch를 줄이거나,
  `make_subset.py --no-include-all`로 끄고 예산 안에서만 뽑는다.

### stairs: AI-Hub Surface 데이터의 caution_zone

stairs는 Polygon 데이터에 **0개**라 Surface(인도 보행 Surface masking) 데이터가 있어야 학습된다.
`prepare_dataset.py --extra-src <Surface 폴더>`가 CVAT XML에서 `caution_zone` 라벨 + `stairs` 속성(`caution_zone/stairs`)인 폴리곤만 골라
**계단이 든 사진만** 추가한다. 나머지 Surface 라벨(sidewalk, manhole 등)은 버린다.

```bash
SURFACE=data/raw_surface bash scripts/run_pilot.sh        # 또는
python tools/prepare_dataset.py --src data/raw --out data/processed --extra-src data/raw_surface --extra-keep stairs
```

- Surface 사진 속 차·사람·기둥은 라벨이 없어서 배경으로 학습된다. 계단 사진만 넣는 이유이고, 그래도 수가 많으면 다른 클래스 성능이 떨어질 수 있다.
- Surface 라벨이 XML이 아니라 마스크 PNG이거나, 계단 표기가 다르면(`classes.yaml`의 `aliases`에 추가) 0개로 나온다.
  `prepare_report.txt`의 stairs 개수와 버린 라벨 목록을 먼저 확인할 것.
- 이미 만든 `index.csv`에는 stairs 열이 없다. Surface를 붙일 때는 `data/processed`를 지우고 다시 만든다.

## 데이터 선택: 랜덤 대신 클래스 균형 + 시퀀스 단위

분포가 매우 치우쳐 있다(obstacle 29.5만 ↔ scooter 351). 무작위로 뽑으면 파일럿 부분집합에 scooter가 거의 들어가지 않는다.
또 영상 프레임이라 인접 프레임이 거의 같은 이미지다. 그래서 `tools/make_subset.py`는 다음 순서로 뽑는다.
균형은 **원본 25종**(제외한 traffic_sign 빼고) 기준으로 맞춘다. 그래야 other_vehicle 안의 wheelchair·stroller, obstacle 안의 kiosk·table처럼
합쳐진 클래스 속 희귀 원본 클래스도 골고루 들어간다. 모델은 10클래스 라벨로 학습한다.

1. **시퀀스(폴더) 단위 train/val/test 분할** (기본 75/15/10). 희귀 클래스부터 배치하는 iterative stratification이라
   scooter·wheelchair·stroller도 모든 split에 비율대로 들어가고, 같은 영상의 프레임이 split을 넘나들지 않는다.
   분할 결과는 `splits/groups.json`에 저장되며 **파일럿과 전체 실험이 같은 분할을 쓴다** (test 시퀀스는 끝까지 보지 않는다).
2. **필수 사진**: `include_all` 클래스가 든 사진을 전부 먼저 넣는다(위 참고).
3. **quota 단계**: 희귀 클래스부터 원본 클래스마다 최소 `--min-train-per-class`(기본 300)장이 될 때까지 해당 클래스가 있는 이미지를 고른다.
   300장보다 적은 클래스(scooter 224장, wheelchair 176장)는 **전부** 들어간다. 후보는 시간축으로 고르게 솎고, 같은 시퀀스에서 많이 뽑을수록 점수를 깎는다.
4. **fill 단계**: 남은 예산은 repeat-factor 가중치(`max_c sqrt(t / f_c)`)로 뽑아서 흔한 클래스만 있는 이미지 비중을 줄인다.
5. **RFS**(LVIS repeat-factor sampling, `--rfs-t 0.1`): 희귀 클래스가 있는 train 이미지를 `train.txt`에 여러 번 적는다. 모든 실험에 똑같이 적용된다.

결과는 `subset_report.md`에 **10클래스 표**와 **원본 클래스 표**로 남는다(전체 / train / val / test / RFS 적용 후).
실제 규모(9만 장, 1,900 시퀀스)의 가짜 인덱스로 돌려 보면 약 1초가 걸리고, scooter는 이미지 235장이 모두 쓰인다(train 172 / val 38 / test 25, RFS 후 train 인스턴스 587).

## 몬드리안 AI 서버에서 실행

```bash
git clone <this repo> && cd <repo>
bash scripts/setup_mondrian.sh                 # GPU·CUDA 확인 + 패키지 설치 + GPU에서 학습 경로 점검 + yolo11s-seg.pt 받기
export HF_TOKEN=hf_xxx                         # DINOv3는 gated → HF 페이지에서 라이선스 동의 후 토큰 발급
python tools/check_teachers.py                 # SigLIP2 / DINOv3 / RADIO / C-RADIO를 받아 GPU에서 실행해 보기

# 1) 구글 드라이브 데이터 받기 ("링크가 있는 모든 사용자" 공유, zip/tar 권장 — gdown은 폴더당 50개 파일 제한)
GDRIVE_URL="https://drive.google.com/file/d/<id>/view" bash scripts/download_data.sh     # → data/raw

# 2) 파일럿 전체 (전처리 → 균형 샘플링 → teacher 점검 → E0,B1..B5,A-best → 요약)
nohup bash scripts/run_pilot.sh > pilot.log 2>&1 &
tail -f pilot.log
```

`run_pilot.sh` 환경 변수: `RAW`, `SURFACE`(stairs용 Surface 폴더), `DATA`, `PROJECT`, `TRAIN_IMAGES`(기본 10000), `VAL_IMAGES`(2500), `DEVICE`(0),
`EXTRA`(예: `"epochs=40 batch=32"`), `OPTIONAL=1`(B2b·B5b 추가).
중간에 끊겨도 다시 실행하면 `results.json`이 있는 실험은 건너뛴다. 실험마다 별도 프로세스라 하나가 실패해도 다음 실험은 계속 돈다.

- 데이터가 이미 서버에 있으면 `RAW=/path/to/data`만 지정한다. 입력 형식은 자동으로 감지한다.
  - **CVAT XML** (AI-Hub 인도보행 Polygon 원본: 시퀀스 폴더 + `<image><polygon label points>` XML)
  - **YOLO-seg** (`.../images/*.jpg` + `.../labels/*.txt`). 이 경우 시퀀스는 이미지의 상위 폴더 이름,
    또는 `python tools/prepare_dataset.py --group-regex '^(.*)_\d+$'` 처럼 파일명에서 뽑는다.
    YOLO 라벨은 **원본 26종 id**(`source_names` 순서)여야 한다. 순서가 다르면 `--src-names <원본 data.yaml>`을 주면 이름으로 다시 매핑한다.
- 원본 이미지는 복사하지 않고 심볼릭 링크를 건다(`--link copy`로 변경 가능).
- 한 실험만 돌리기: `python tools/train.py --exp B2 --data data/processed/splits/pilot/data.yaml --project runs/pilot`
- A-best teacher를 직접 지정: `python tools/train.py --exp A-best --teacher dinov3_b ...`

### 결과 보기

```
runs/pilot/summary_val.md      # 실험별 mask/box mAP, E0 대비 Δ, key / rare / focus mAP, scooter·traffic_light AP, 추론 ms, 학습 시간
runs/pilot/per_class_val.csv   # 클래스 × 실험 mask mAP50-95
runs/pilot/<ID>/results.json   # 실험별 상세 결과
runs/pilot/<ID>/weights/       # best.pt (Distill은 배포용 student.pt 추가)
runs/pilot/logs/<ID>.log
```

**선택 기준**: val **mask mAP50-95**가 1순위이고, key mAP(보행 안전 핵심 클래스)와 rare mAP(scooter 등)를 함께 본다.
A-best는 배포 후보가 아니라 증류로 얼마나 가까이 따라갔는지 비교하는 기준선이다.

## 전체 실험 (파일럿 다음 단계)

```bash
EXPS="E0 B2" nohup bash scripts/run_full.sh > full.log 2>&1 &    # B2 자리에 파일럿 1등 ID
```

같은 시퀀스 분할에서 train/val 전체 이미지를 쓰고(`splits/full`), 기본값 `epochs=150 patience=40`으로 학습한 뒤
처음으로 **test 시퀀스**까지 평가한다(`runs/full/summary_test.md`).

## 기본 하이퍼파라미터 (`configs/experiments.yaml`)

| 항목 | 값 | 비고 |
|---|---|---|
| student | `yolo11s-seg.pt` (COCO 사전학습) | Ultralytics 8.4.166 고정 |
| imgsz / batch / epochs | 640 / 16 / 60 (patience 20) | `--set` 또는 `EXTRA`로 변경 |
| KD 층 | 4, 6, 10 (P3/P4/P5, stride 8/16/32) | |
| KD 손실 | `1 - cosine`, weight 1.0 | `loss: mse` 선택 가능 |
| teacher 입력 | 긴 변 512 px | DINOv2는 patch 14 → 37×37 |

모든 실험은 seed 0, 같은 데이터 목록, 같은 증강을 쓰고 teacher만 바뀐다.

## 구조

```
configs/   classes.yaml (원본 26종 → 학습 10클래스 매핑, key/rare), experiments.yaml (E0/B1–B5/A-best)
kdseg/     teachers.py (SigLIP2 / DINOv3 / DINOv2 / RADIO 래퍼), models.py (DistillSegModel, FusionSegModel, student 추출)
tools/     prepare_dataset.py → make_subset.py → train.py / run_experiments.py → summarize.py, check_teachers.py
scripts/   setup_mondrian.sh, download_data.sh, run_pilot.sh, run_full.sh
tests/     make_synthetic.py, smoke_test.sh (합성 데이터 + 랜덤 teacher로 CPU에서 전체 파이프라인 확인)
```

## 주의 사항

- **GPU는 1장 기준**이다. teacher를 모델 밖(프로세스 전역)에 두기 때문에 Ultralytics의 멀티 GPU(DDP) 모드는 지원하지 않는다.
- **RADIO/C-RADIO**는 `torch.hub`(GitHub `NVlabs/RADIO`)에서 먼저 받고, 실패하면 Hugging Face(`nvidia/C-RADIOv3-B` 등)로 받는다. 서버가 GitHub과 HF에 접속할 수 있어야 한다.
- 긴 학습 전에 `python tools/check_teachers.py`로 모든 teacher가 받아지고 돌아가는지 먼저 확인한다(`run_pilot.sh`는 이 단계를 자동으로 실행한다).
- **GPU·CUDA 점검**: `python tools/check_gpu.py`는 torch·CUDA·cuDNN 버전, GPU 이름·메모리, bf16 지원 여부를 출력하고,
  증류·Fusion 모델을 GPU에서 AMP로 한 번 학습해 본다(다운로드 없음, 1분 이내). `setup_mondrian.sh`와 `run_pilot.sh`가 자동으로 실행한다.
  torch가 CPU 전용이면 `setup_mondrian.sh`가 `nvidia-smi`의 드라이버 CUDA 버전에 맞는 torch를 설치한다. 이미 GPU용 torch가 있으면 건드리지 않는다.
- teacher는 bf16을 지원하는 GPU(A100, H100, RTX 30·40 계열 등)에서는 bf16으로, 아니면(V100, T4 등) fp16으로 돌아간다.
- 검증 범위: 합성 데이터와 랜덤 teacher로 데이터 변환 → 샘플링 → E0/Distill/Fusion 학습 → student 추출 → 평가 → 요약까지 **CPU에서** 통과했다(`bash tests/smoke_test.sh`).
  SigLIP2·DINOv3·DINOv2 래퍼는 같은 구조의 작은 랜덤 모델로 토큰 → feature map 변환 모양을 확인했다.
  **실제 GPU 실행과 실제 teacher 가중치(SigLIP2, DINOv3, RADIO, C-RADIO) 다운로드는 개발 환경에 GPU가 없고 Hugging Face·GitHub 접속이 막혀 있어 확인하지 못했다.**
  서버에서 `check_gpu.py` → `check_teachers.py` 순서로 먼저 확인할 것.
