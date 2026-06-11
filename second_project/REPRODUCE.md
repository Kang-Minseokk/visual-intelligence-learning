# 재현 방법 — DINO ViT-S/8 on STL-10

```bash
git clone <repo-url>
cd visual-intelligence-learning/second_project
pip install -r requirements.txt
```

그 다음 학습 + 평가 (약 14시간 소요):

```bash
# 포그라운드 (터미널 닫으면 죽음)
bash reproduce_best.sh

# 또는 백그라운드 (터미널 닫아도 OK, 권장)
nohup bash reproduce_best.sh > reproduce.log 2>&1 &
disown

# 진행 상황 확인 (백그라운드일 때)
tail -f reproduce.log
# 또는 epoch별 학습 곡선:
tail -f output/dino_vits8_sk_v2_fresh_stl_stats/log.csv
```

약 14시간 후 `reproduce.log` 마지막에 표시:

```
Final Results
stl10      Top-1: ~93.2%
cifar10    Top-1: ~89.0%
```

저장 위치: `output/dino_vits8_sk_v2_fresh_stl_stats/eval_results.txt`

요구사항: NVIDIA GPU (RTX 30xx/40xx, A100 등 Ampere+, VRAM 14GB+), 디스크 10GB, 인터넷.
