from ultralytics import YOLO

model = YOLO("yolo26s.pt") # qué YOLO se usa
epochs = 50
#'''
model.train(
    data="dataset_yolo_corners/data.yaml",
    epochs=epochs,
    imgsz=320, #input image size, tiene que ser múltiple de 32
    name=f"exp1_only_kaggle_YOLO26_corners_epochs{epochs}",
    device="cpu",
    patience = 20, # patience permite acabar antes el entrenamiento. Cantidad de epochs sin cambios antes de parar
    save=True,
)
# '''
# Para cuando falla

# model = YOLO("runs\detect\exp1_only_kaggle_YOLO26_epochs21\weights\last.pt") # REFERENCIAR EL MODELO A CONTINUAR
# model.train(resume=True)
