# demo_platform

demo에 필요한 package를 설치합니다.

```
pip install -U pip   
pip install tensorflow matplotlib pillow numpy kfp
```

## notebook에서 학습 진행
notebook에서 필요한 패키지를 설치한 후 mnist 모델을 학습합니다.
```commandline
python mnist.py
```
학습 후 tensorflow 모델을 생성하며 학습된 모델은 예측값을 파일명으로 생성합니다. ex) 0~9.jpg

## hyperparameter tuning 진행
하이퍼파라미터 튜닝을 위한 코드는 ```mnist_hyper_tuning.py```입니다.   
해당 코드를 ```하이퍼파라미터 튜닝```탭에서 사용하세요   

| `하이퍼파라미터`    | `type`   | `min` | `max`  | `step` |
|--------------------|--------|------|------|----|
| num_hidden_layer_1 | integer | 64   | 256  |64|
| num_hidden_layer_2 | integer | 64   | 256  |64|
| dropout | double | 0.25 | 1.0  | 0.25 |
| learning_rate | double | 0.01 | 0.1  | 0.01 |
| epoch | integer | 1000  | 3000 |500|

## pipeline 작성
mnist 학습을 진행하는 워크 플로우를 한눈에 보기 쉽도록 합니다.
데이터 로드 및 학습, 예측 까지의 모든 과정을 한번에 확인할 수 있습니다.
```commandline
python mnist_pipeline.py
```
실행을 하면 `.yaml`파일이 생성됩니다.   
```파이프라인``` 탭에서 생성된 파일을 실행하세요

## model serving
모델을 학습하였으면 직접 사용하는 기능을 제공합니다.   
`mnist.py`를 수행하면 `mnist_model` 디렉토리가 생성됩니다. 생성된 디렉토리는 학습된 tensorflow mnist 모델입니다.   
```모델 서빙```탭에서 생성된 모델을 사용하여 `endpoint`를 얻으세요   
획득한 `endpoint`는 `mnist_model_serv.ipynb`에서 `URL`에 담아준 후 실행합니다.   
   
다른 이미지를 사용하려면 0~9의 숫자를 선택합니다.