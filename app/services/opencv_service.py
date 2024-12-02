"""Carregamento do OPENCV"""
import threading  # Usando threading em vez de multiprocessing

import cv2

from app.services.custom_exceptions import cameraDoesntOpenExcpetion
from app.services.yolo_service import yoloModel
from app.services.controls_service import Button

class OpencvService:
    """Este é o service do opencv"""
    def __init__(self, yolo_service: yoloModel):
        """Instanciação do objeto, este objeto serve para encapsular a execução do opencv visando modularidade do código

        Args:
            yolo_service (yoloModel): a service do yolo para que seja feita a previsão do botão
        """
        self.yolo = yolo_service
        self.thread = threading.Thread(target=self.do_detection_on_cap)
        self.thread.start()

    def do_detection_on_cap(self):
        """Faz a detecção na captura de imagem do opencv

        Raises:
            cameraDoesntOpenExcpetion: Caso ocorra algum problema com a câmera do usuário, esta exceção será levantada
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            raise cameraDoesntOpenExcpetion(
                "Ops, alguma coisa aconteceu e não foi possível abrir sua câmera, por favor tente outra engine que não seja o MSMF"
            )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha na leitura do frame da câmera.")
                break
            
            predicted_frame, detection_classes = self.yolo.predict(frame)
            
            for class_name in detection_classes:
                button = Button(class_name)
                print(f"Classe detectada: {class_name}, Botão criado: {button}")

            # Exibe o resultado
            cv2.imshow("Predictions", predicted_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Encerrando o programa.")
                break
        
        # Libera a câmera e fecha as janelas após o loop
        cap.release()
        cv2.destroyAllWindows()
