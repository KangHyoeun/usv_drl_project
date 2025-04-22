# USV Deep Reinforcement Learning Collision Avoidance

[![GitHub stars](https://img.shields.io/github/stars/KangHyoeun/usv_drl_project.svg?style=social)](https://github.com/KangHyoeun/usv_drl_project)

🚤 **Deep Reinforcement Learning-based Collision Avoidance for Unmanned Surface Vehicles (USVs)**  
본 프로젝트는 우주현(2018)의 석사 논문 *「심층강화학습을 이용한 무인수상선의 충돌회피」* 내용을 기반으로,  
USV의 충돌 회피 알고리즘을 강화학습으로 학습하고 시뮬레이션하는 프레임워크입니다.

---

## 🧠 주요 특징

- ✅ **NED 좌표계 통일** (X=북, Y=동, ψ=0=북쪽)
- ✅ **Dueling DQN + Target Network 기반 강화학습**
- ✅ **Tam & Bucknall 방식 조우 유형 분류**
- ✅ **Head-on, Give-way, Overtaking, Stand-on 시나리오 지원**
- ✅ **Otter USV (L=2.0m) 기반 동역학 모델 적용**
- ✅ **조우 상황별 조건 필터링 기반 장애물 spawn**
- ✅ **Grid map 시각화, reward curve 로깅, trajectory 추적 지원**

---

## 📦 프로젝트 구조

```
usv_drl_project/
├── envs/
│   └── usv_collision_env.py       # 강화학습 환경 (Gym 호환)
├── models/
│   └── dueling_dqn.py             # Dueling DQN 모델 구조
├── utils/
│   ├── guidance.py                # 경로 추종 / 회피 유도기
│   ├── control.py                 # PID + 제어 할당
│   ├── cpa_utils.py               # CPA/DCPA 계산
│   ├── encounter_classifier.py    # 조우유형 분류
│   ├── gridmap_utils.py           # Grid map 생성
│   ├── reward_utils.py            # 보상 함수
│   └── replay_buffer.py           # 경험 리플레이 버퍼
├── train.py                       # 학습 메인 스크립트
├── test.py                        # 학습된 정책 시각화
├── config.py                      # 파라미터 설정
└── README.md
```

---

## 🚀 설치 및 실행

```bash
git clone https://github.com/KangHyoeun/usv_drl_project.git
cd usv_drl_project
pip install -r requirements.txt

# 학습 시작
python train.py
```

---

## 🧪 시나리오 기반 학습 (config.py 설정)

```python
# config.py
'scenario': 'HO'  # Head-on
# 또는 'GW', 'SO', 'OT'로 변경하여 조우 유형 전환 가능
```

---

## 📊 학습 결과 분석

- `/logs/train_*.csv` → reward / loss curve 확인 가능
- `render()` → 자선과 장애물 움직임 시각화
- trajectory 저장 및 회피 행동 시퀀스 시각화 가능

---

## 📖 참고 논문

우주현, 2018.  
**「심층강화학습을 이용한 무인수상선의 충돌회피」**  
서울대학교 대학원 조선해양공학과 석사학위논문.

---

## 📬 문의

- Maintainer: [@KangHyoeun](https://github.com/KangHyoeun)
- 질문 및 제안: [Issues](https://github.com/KangHyoeun/usv_drl_project/issues) 탭 이용
