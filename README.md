# 🖊️ Handwritten Digit Recognition with Neural Network
손글씨 숫자 인식을 위한 신경망 구현 (Python &amp; Numpy)

## 📌 프로젝트 개요
이 프로젝트는 손글씨 숫자 인식 (Handwritten Digit Recognition)을 위한 다층 퍼셉트론 신경망 (MLPClassifier)을 활용하여 **MNIST 스타일의 데이터를 학습하고 예측하는 모델**을 개발한 것입니다.  
Google Colab에서 처음 작성되었으며, 이후 VS Code 환경에서도 실행할 수 있도록 조정되었습니다.

---

## 🕒 개발 히스토리
- 📌 **최초 작성:** 2023년 12월 12일 (Google Colab)  
- 📌 **업데이트:** 2025년 3월 (VS Code 환경에서 실행 가능하도록 수정)  
- 📌 **Google Drive에서 저장 → GitHub로 관리하도록 변경**  
- 📌 **Colab에서 실행한 학습 과정 캡처(GIF) 포함 (`gscv2.gif`, `gscv2_2.gif`)**  

---

## 📂 프로젝트 폴더 구조
```
📂 dataset                 # 데이터 관련 폴더
 ├── trainlabels.bin       # 학습 데이터 라벨
 ├── trainimages.bin       # 학습 데이터 이미지
 ├── testlabels.bin        # 테스트 데이터 라벨
 ├── testimages.bin        # 테스트 데이터 이미지
 ├── test.png              # 손글씨 숫자 5 이미지 (예측 테스트용)
 ├── test.bin              # 테스트 이미지의 바이너리 데이터
 ├── sample_images/        # 직접 그린 손글씨 숫자 이미지 (0~9)
 │    ├── 0/  (0번 손글씨 숫자 4개)
 │    ├── 1/  (1번 손글씨 숫자 4개)
 │    ├── ...  
 │    ├── 9/  (9번 손글씨 숫자 4개)

📂 model                   # 모델 학습 & 결과 저장 폴더
 ├── learningdata.npz      # 학습된 가중치 저장

📂 scripts                 # 코드 관련 폴더
 ├── learning_book.py      # 신경망 학습 코드 (Colab에서 작성 후 VS Code에서 실행 가능하도록 수정됨)
 ├── predict_book.py       # 손글씨 숫자 예측 코드
 ├── accuracy_check.py     # 전체 테스트 데이터셋 정확도 계산
 ├── predict_custom_1.py   # 특정 테스트 이미지에 대한 예측 및 시각화
 ├── predict_custom_2.py   # 직접 그린 손글씨 숫자 예측
📄 README.md               # 프로젝트 설명 파일
```

---

## 🛠 실행 방법
### 📌 1️⃣ 모델 학습
```bash
python scripts/learning_book.py
```

### 📌 2️⃣ 손글씨 숫자 예측
```bash
python scripts/predict_book.py
```

### 📌 3️⃣ 전체 테스트 데이터셋 정확도 확인
```bash
python scripts/accuracy_check.py
```

### 📌 4️⃣ 특정 테스트 이미지 시각화 및 예측
```bash
python scripts/predict_custom_1.py
```

### 📌 5️⃣ 직접 그린 손글씨 숫자 예측
```bash
python scripts/predict_custom_2.py
```

---

## 📊 Google Colab
| Google Colab_1 | Google Colab_2 |
|-----------|---------------|
| ![Training Process](model/gscv2_2.gif) | ![Hyperparameter Tuning](model/gscv2.gif) |

---

## 🎯 최종 결과 (VS Code에서 실행된 결과)
- 🔹 **테스트 데이터셋 정확도:** 96.15%
- 🔹 **test.png 예측 결과:** 5 (모델이 정확히 예측함)
- 🔹 **특정 테스트 이미지 예측 결과:** 6 (예: index=11인 이미지)
- 🔹 **사용자 손글씨(0_1.png) 예측 결과:** 0 (모델이 잘 인식함)

---

## 🏠 마무리
📌 Google Colab에서 처음 개발된 모델로, 이후 VS Code 환경에서도 실행 가능하도록 변환하여 보다 편리하게 활용할 수 있도록 조정되었습니다. 모델의 예측 성능을 검증할 수 있도록 다양한 테스트 스크립트가 추가되었습니다.

