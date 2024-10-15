import cv2
import numpy as np


# Cargar los nombres de las clases y la configuración del modelo YOLO
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("objetos.txt", "r") as f:  # Asegúrate de tener un archivo con los nombres de las plantas
        classes = [line.strip() for line in f.readlines()]
        print(classes)
    return net, classes

# Dibuja los cuadros delimitadores y la etiqueta de clase en la imagen
def draw_bounding_boxes(frame, boxes, confidences, class_ids, classes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Función principal
def main():
    cap = cv2.VideoCapture(0)
    net, classes = load_yolo_model()

    # Obtener nombres de capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    while True:
        
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        print(outs)
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]  # Obtén las puntuaciones de las clases
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Ajusta este umbral según lo necesario
                if confidence > 0.6 and class_id < len(classes):  # Umbral de confianza
                    if classes[class_id] in classes:  # Asegúrate de que sea una planta
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Coordenadas de los cuadros delimitadores
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # Supresión de no máximos para eliminar duplicados
        # Ajusta el segundo parámetro (0.5) si es necesario
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.3)  # Reduce el umbral NMS a 0.3

        if len(indices) > 0:
            for i in indices.flatten():
                # Dibuja solo los cuadros que pasaron por NMS
                box = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                draw_bounding_boxes(frame, [box], [confidence], [class_id], classes)

        cv2.imshow('Camera', frame)

 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
