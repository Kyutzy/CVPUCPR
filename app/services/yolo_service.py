"""Este arquivo é o serviço da yolo"""
import glob
import re

from typing import Union

from ultralytics import YOLO
from cv2 import Mat
import torch

from app.services.custom_exceptions import modelNeverTrainedBefore

class yoloModel:
    """Esta classe faz a criação e gerenciamento da yolo"""
    def __init__(self, load_last_weights: bool=False) -> None:
        """Inicialização do objeto da yolo

        Args:
            load_last_weights (bool, optional): Se o modelo a ser carregado, é o último modelo treinado. Padrão é False.
        """
        model_path = "yolo11n.pt"
        if load_last_weights:
            last_execution_path = self.find_last_execution()
            model_path = f"{last_execution_path}/weights/best.pt"

        self.model = YOLO(model_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def train(self, yaml_data: str, **kwargs) -> None:
        """Faz o treinamento da yolo com o argumento yaml_data

        Args:
            yaml_data (str): o caminho do arquivo yaml com os dados de treino, validação e teste
        """
        self.model.train(data=yaml_data, **kwargs)

    def predict(self, frame: Mat) -> Union[Mat|list]:
        """Faz a predição utilizando o modelo carregado na instância da classe

        Args:
            frame (Mat): Frame da captura de vídeo do opencv

        Returns:
            Union[Mat|list]: Uma tupla onde o 0 é a imagem anotada pela YOLO e o 1 é uma lista com todas as classes que aparecem na imagem
        """
        result = self.model(frame)

        predicted_frame: Mat = result[0].plot()

        detections = []
        boxes = result[0].boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls)
                class_name = self.model.names[class_id]
                detections.append(class_name)
        else:
            print("Nenhuma detecção encontrada.")

        return predicted_frame, detections

    def find_last_execution(self) -> str:
        """Encontra a última execução gravada em disco

        Raises:
            modelNeverTrainedBefore: Caso o modelo nunca tenha sido treinado, esta exceção é levantada

        Returns:
            str: nome da pasta onde está salvo o peso do último treino do modelo
        """
        folders = glob.glob('runs/detect/*')
        if len(folders) == 0:
            raise modelNeverTrainedBefore("Este modelo nunca foi treinado antes, logo, não é possível carregar pesos anteriores, treine o modelo antes.")

        sorted_folders = sorted(
            folders,
            key=lambda path: int(re.search(r'\d+', path).group()) if re.search(r'\d+', path) else float('0')
        )
        return sorted_folders[-1]
