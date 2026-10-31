import os
import csv
import shutil
import cv2
# EJECUTARLO DE NUEVO SOBREESCRIBE LOS ARCHIVOS
# Antes de ejecutar, hay que crear las carpetas:
# dataset
#   - images
#       test, train & valid
#   - labels
#       test, train & valid

# CODIFICAR LAS CLASES
# Orden de los valores
numeros_string = [
    "ace", "two", "three", "four", "five",
    "six", "seven","eight","nine","ten",
    "jack","queen","king" ]

# Orden de los palos
palos = [ "clubs", "diamonds", "hearts", "spades"]
print(palos[0])

# Crear las clases en el orden deseado
class_names = []

for palo in palos: # palo a palo
    for value in numeros_string: # Numero a numero
        # Creamos una clase en la lista con el nombre igual al de las carpetas que ya tenemos
        class_names.append(f"{value} of {palo}")

# Joker
class_names.append("joker")

# Mostrar las clases y sus IDs
# enumerate devuelve el índice (class_id) y el elemento (class_name)
for class_id, class_name in enumerate(class_names):
    print(class_id, "->", class_name)

# Quiero guardar esto en un csv. # Mismas clases siempre así que lo dejo comentado
'''
with open("clases.csv", "w", newline="", encoding="utf-8") as archivo:
    escritor = csv.writer(archivo)

    escritor.writerow(["ID", "Clase","Numero","Palo (num)","Palo (str)"]) # Cabecera

    for class_id, class_name in enumerate(class_names):
        palo_num = class_id//13 #El palo expresado como numero
        if palo_num <=3:
            palo_str = palos[palo_num] # El palo en letra
            carta_num = class_id - palo_num * 13 +1  # Numero de la carta
        else: # es el joker
            palo_num = -1
            palo_str = "Joker"
            carta_num = -1
        escritor.writerow([class_id, class_name, carta_num, palo_num, palo_str]) #Escribimos en el csv
# También así podemos trabajar en pandas, si cal...
'''

# Crear un diccionario nombre_de_clase -> ID
class_to_id = {
    class_name: class_id
    for class_id, class_name in enumerate(class_names)
}

# TRASLADO DE ARCHIVOS
dataset_path = r"C:\Users\maria\Julianadas\RealBalatro\data" #r = raw string, pq \ es para los caracteres especiales
yolo_path = r"C:\Users\maria\Julianadas\RealBalatro\dataset_yolo_corners"
# For the bounding boxes
x_c = 0.1 # xcenter
y_c = 0.15 #ycenter
x_l = 0.2 #xlength = width
y_l = 0.3 #ylength = height


for split in ["train", "valid", "test"]: #Visita las tres carpetas

    split_path = os.path.join(dataset_path, split)

    for class_name in os.listdir(split_path): #Visita las carpetas de cada tipo de carta

        class_path = os.path.join(split_path, class_name) # Path

        if not os.path.isdir(class_path): # Si no lo encuentra
            continue

        # Comprobar que la clase existe en nuestro listado
        if class_name not in class_to_id:
            print(f"AVISO: clase no reconocida: {class_name}")
            continue

        class_id = class_to_id[class_name] #Mapar de vuelta

        # Recorrer imágenes
        for filename in os.listdir(class_path):
            # Busca las fotos
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Rutas a la foto
            source = os.path.join(class_path, filename)

            # Para evitar problemas si dos clases tienen imágenes con el mismo nombre,
            # añadimos el ID de clase al nombre.
            new_filename = f"{class_id}_{filename}"

            destination_image = os.path.join(
                yolo_path, # EN la carpeta del dataset del yolo
                "images", # En la carpeta de imágenes
                split, #en la subcarpeta que hemos escogido (train/test/validation)
                new_filename #El nuevo archivo
            )

            destination_label = os.path.join(
                yolo_path, # EN la carpeta del dataset del yolo
                "labels", # En la carpeta de etiquetas
                split, #en la subcarpeta que hemos escogido (train/test/validation)
                os.path.splitext(new_filename)[0] + ".txt"  #El nuevo archivo
            )
            print(f"Source: {source}")
            print(f"Destination image: {destination_image}")
            print(f"Destination label: {destination_label}")

            # Copiar imagen
            shutil.copy2(source, destination_image)

            # Como cada imagen contiene una única carta y la carta ocupa toda la imagen:
            # class_id x_center y_center width height
            label = f"{class_id} {x_c} {y_c} {x_l} {y_l}\n"

            with open(destination_label, "w") as f:
                f.write(label)

print("Conversión terminada.")

# Para comprobar que se ha ejecutado correctamente
for split in ["train", "valid", "test"]:
    images_path = os.path.join(yolo_path, "images", split)
    print(images_path)
    labels_path = os.path.join(yolo_path, "labels", split)
    # Printear la bounding box y guardar en carpeta extra
    ''' 
    check_path = r"C:\ Users\maria\Julianadas\RealBalatro\dataset_yolo_corners\check"
    corner_up_left = (200*(x_c-x_l/2), 200*(y_c-y_l/2))
    corner_down_right = (200*(x_c+x_l/2), 200*(y_c+y_l/2))
    print(f"Top, bot corners: {corner_up_left}, {corner_down_right}:")
    img = cv2.imread(os.path.join(images_path))
    cv2.rectangle(img,(0, 0),(40,60),(0,255,0),2)
    cv2.imshow("img",img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    '''
    #Contar imagenes y etiquetas
    n_images = len(os.listdir(images_path))
    n_labels = len(os.listdir(labels_path))
    print(
        f"{split}: "
        f"{n_images} imágenes |"
        f"{n_labels} etiquetas"
    )

# ARCHIVO YAML
# para que el YOLO sepa donde encontrar sus fotos)
yaml_path = os.path.join(yolo_path, "data.yaml")

with open(yaml_path, "w") as f:

    f.write(f"path: {yolo_path} \n")
    f.write("train: images/train\n")
    f.write("val: images/valid\n")
    f.write("test: images/test\n\n")

    f.write(f"nc: {len(class_names)}\n")

    f.write("names:\n")

    for class_id, class_name in enumerate(class_names):
        f.write(f"  {class_id}: {class_name}\n")

print("data.yaml creado en:")
print(yaml_path)