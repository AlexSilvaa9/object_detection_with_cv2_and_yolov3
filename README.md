# Natural Park Surveillance Project using YOLO

This project is designed to aid a group of Environmental Science students at the University of Málaga (UMA) in developing a surveillance system for natural parks. The project leverages YOLO (You Only Look Once), a popular object detection model, to monitor and identify various plant species and other relevant objects within natural park areas. The system captures video input and uses pre-trained YOLO models to detect objects of interest, displaying bounding boxes and confidence levels for each detected object.

## Project Purpose

The main goal of this project is to assist in the environmental monitoring of natural parks. By using advanced computer vision techniques, we aim to:

- Identify and monitor various plant species in real-time.
- Detect potential anomalies or unwanted activities within the park.
- Provide data that can be used to further the protection and conservation efforts within natural park areas.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- YOLOv3 weights and configuration files:
  - `yolov3.weights`
  - `yolov3.cfg`
- A text file (`objetos.txt`) with the names of the plant species to detect.

## How to Run

1. Ensure that you have all the necessary Python libraries installed:
    ```
    pip install opencv-python numpy
    ```

2. Download the YOLOv3 model weights and configuration files from [YOLO's official website](https://pjreddie.com/darknet/yolo/).

3. Place the `objetos.txt` file in the same directory as the script, ensuring it contains the names of the objects or plant species you wish to detect.

4. Run the Python script:
    ```
    python surveillance.py
    ```

5. The script will open your webcam and start detecting objects based on the loaded YOLO model.

6. Press `q` to stop the camera feed and exit the program.

## Code Overview

- **load_yolo_model()**: Loads the YOLO model and object classes from the provided files.
- **draw_bounding_boxes()**: Draws the bounding boxes and labels for detected objects on the video feed.
- **main()**: Captures video input, processes each frame using YOLO, and displays the results in real-time.

## Future Improvements

- Expanding the model to detect additional plant species or objects.
- Implementing cloud-based storage for the detected data for further analysis.
- Enhancing the detection model to reduce false positives and improve accuracy in various environmental conditions.

## Contributions

This project is part of a collaborative effort by students in the Environmental Science program at UMA, aiming to combine AI and conservation technology to protect natural parks.
