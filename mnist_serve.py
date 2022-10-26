import requests
import os
import json
import base64
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
"""
Image Classification Tutorial : MNIST 손글씨
출처 : TorchServe Github : https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist
Pytorch 기반의 TorchServe는 하나의 내부 IP Address에 Endpoint가 모델 단위로 분류됩니다.
모델의 목록을 조회하고 싶다면, [GET] http://${IP_ADDRESS}:80/v1/models 를 통해 확인할 수 있습니다.
"""
# 모델 서버 예측 REST API
# Model Name : mnist
# URL = {REST_API_URL}


def main(image_num, model_endpoint):
    # 테스트 이미지 경로
    image_path = f'{os.getcwd()}/test_data/{image_num}.jpg'
    img = Image.open(image_path).convert("L")
    img = np.array(img)
    img = img.reshape([-1, 784])


    # KServe Input 포맷에 맞추는 작업
    input_data = {"instances": img.tolist()}

    print("\n##### Input Data #####")
    print(input_data)

    # JSON으로 Input 포맷 변환
    data = json.dumps(input_data)

    # 모델 예측 API 호출 및 응답 저장
    prediction = requests.post(model_endpoint, data)

    result = list(str(prediction.json()["predictions"][0])[1:-1].split(','))
    prediction_value = result.index(str(' 1.0'))

    print("\n##### 예측 및 분류 결과 #####")
    print(f"---------- 예측 숫자 : {prediction_value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image_number', type=int, default =1, help='input int 0~9')
    parser.add_argument('--model_endpoint', type=str, default =None, help='input_model_dir')
    args = parser.parse_args()

    main(args.test_image_number, args.model_endpoint)


