# network — 신경망 (CNN 혼합: 2D 관측마다 CNN + MLP)

| 파일 | 내용 |
|---|---|
| `conv.cu` | 합성곱 층 앞·뒤 계산 |
| `mlp.h` · `mlp.cu` | 완전연결 층 앞·뒤 계산 |
| `distribution.cu` | 행동 분포: 가우시안(v, ω) + 베르누이(끝 선언) 샘플·log 확률·엔트로피 |
| `actor_critic.h` · `actor_critic.cu` | 정책·가치 조립. 입력 크기는 `mdp/observations.h` 상수 |
