# USV Deep Reinforcement Learning Collision Avoidance

[![GitHub stars](https://img.shields.io/github/stars/KangHyoeun/usv_drl_project.svg?style=social)](https://github.com/KangHyoeun/usv_drl_project)

🚤 **Deep Reinforcement Learning-based Collision Avoidance for Unmanned Surface Vehicles (USVs)**  
본 프로젝트는 우주현(2018)의 석사 논문 *「심층강화학습을 이용한 무인수상선의 충돌회피」* 내용을 기반으로,  
USV의 충돌 회피 알고리즘을 강화학습으로 학습하고 시뮬레이션하는 프레임워크입니다.

---

## 🧠 주요 특징

- ✅ **NED 좌표계 통일 (X=북, Y=동, ψ=0=북쪽)**
- ✅ **Dueling DQN + Target Network 기반 강화학습**
- ✅ **Tam & Bucknall 방식 조우 유형 분류**
- ✅ **Head-on, Give-way, Overtaking, Stand-on 시나리오 지원**
- ✅ **Otter USV (L=2.0m) 기반 동역학 모델 적용**
- ✅ **각 시나리오에 맞는 장애물 조건 필터링 기반 spawn**
- ✅ **grid map 시각화, reward curve 로깅, 학습 trajectory 추적**

---

## 📦 프로젝트 구조

usv_drl_project/ 
├── envs/ 
│ └── usv_collision_env.py # 강화학습용 환경 (Gym 호환) 
├── models/ 
│ └── dueling_dqn.py # Dueling DQN 구조 
├── utils/ 
│ ├── guidance.py # 유도기 (경로추종 + 회피) 
│ ├── control.py # PID + 프로펠러 제어 
│ ├── cpa_utils.py # CPA/DCPA 계산 
│ ├── encounter_classifier.py # 조우유형 분류기 
│ ├── gridmap_utils.py # Grid map 생성 및 시각화 
│ ├── reward_utils.py # 보상 계산 
│ └── replay_buffer.py # 경험 리플레이 버퍼 
├── train.py # 학습 메인 코드 (병렬 환경 지원) 
├── test.py # 학습된 정책 테스트 
├── config.py # 파라미터 설정 
└── README.md

---

## 🚀 설치 및 실행

```bash
git clone https://github.com/KangHyoeun/usv_drl_project.git
cd usv_drl_project
pip install -r requirements.txt

# 학습 시작
python train.py

---

## 🧪 시나리오 기반 학습 (config.py에서 설정)

'scenario': 'HO'  # Head-on
# 'GW', 'OT', 'SO' 로 변경하여 다른 조우상황 학습 가능
📊 학습 결과 확인
/logs/*.csv 파일로 reward/loss curve 확인

render()를 통해 자선의 회피 기동 시각화 가능

seed sweep, trajectory 기록 등 학습 분석 가능

📖 참고 논문
우주현, 2018.
「심층강화학습을 이용한 무인수상선의 충돌회피」,
서울대학교 대학원 조선해양공학과 석사학위논문.

📬 문의
Repository Maintainer: @KangHyoeun

Issues 또는 Discussions 탭을 통해 질문과 제안 환영합니다!



---

