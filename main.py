from ultralytics import YOLO
import torch

def train_yolo_model():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")


    model_path = 'yolov8n.pt'
    data_yaml_path = r'D:\bagathesh\pythonproject\data.yaml'


    model = YOLO(model_path)

    model.train(
        data=data_yaml_path,
        epochs=25,
        batch=8,
        imgsz=640,
        name='yolo_model_cuda',
        project='runs/train',
        save_period=5,
        patience=10,
        optimizer='AdamW',
        device=device,
        cache='ram',
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005,
    )

if __name__ == "__main__":
    train_yolo_model()

