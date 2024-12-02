"""Este arquivo é a execução principal do programa"""
#pylint: skip-file
from app.services.yolo_service import yoloModel
from app.services.opencv_service import OpencvService
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Descreva o modelo da yolo que você deseja utilizar')
    parser.add_argument('--train', action='store_true', help='Se você deseja treinar o modelo')
    args = parser.parse_args()
    train = args.train
    if not train:
        yolo = yoloModel(load_last_weights=True)
    else:
        yolo = yoloModel()
        yolo.train("data/data.yaml", epochs=100, patience=10)
    OpencvService(yolo)