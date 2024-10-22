from flask import Flask, Response, render_template, request, send_file
from ultralytics import YOLO
import cv2
import torch
import os

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())  # Should return the current device index
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU


# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)
model = YOLO("best.pt").to(device)
# Run YOLO model on the preprocessed frame
model.names

# model.cuda() 




CLASS_NAMES = {
    0: "Person",
    1: "Glasses",
    2: "Gloves",
    3: "Vest",
    4: "Safety Boots",
    5: "Helmet",
    6: "Train",
    7: "Truck",
}


def preprocess_frame(frame):
    # Resize the frame so its width and height are divisible by 32
    height, width = frame.shape[:2]
    new_height = (height // 32) * 32
    new_width = (width // 32) * 32
    resized_frame = cv2.resize(frame, (new_width, new_height))


    # Convert the frame to a tensor and move it to the GPU if available
    tensor_frame = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Convert to BCHW
    tensor_frame = tensor_frame.to(device)  # Move to GPU

    return tensor_frame

# Function to generate video stream
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam

    while True:
        success, frame = cap.read()  # Read a frame from the webcam
        if not success:
            break

        # Preprocess the frame
        tensor_frame = preprocess_frame(frame)

        # Run YOLO model on the preprocessed frame
        results = model(tensor_frame, device=device)


        # Post-process results and draw bounding boxes
        frame = frame_process(frame, results, model)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap = cv2.VideoCapture(0)  # Open the webcam (0 for the default camera)

    while True:
        success, frame = cap.read()  # Read a frame from the webcam
        if not success:
            break
        
        # Run the custom YOLO model on the frame
        results = model(frame)
        
        # Draw bounding boxes on the frame
        for result in results[0].boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Extract the coordinates
            confidence = result.conf[0].item()
            label = f"Class: test, Conf: {confidence:.2f}"

            # Draw the bounding box and label
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# def frame_process(frame, results):
#     # Draw bounding boxes on the original frame
#     for result in results[0].boxes:
#         x1, y1, x2, y2 = map(int, result.xyxy[0])  # Extract the coordinates
#         confidence = result.conf[0].item()
#         label = f"Class: {result.cls[0].item()}, Conf: {confidence:.2f}"

#         # Draw the bounding box and label
#         frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame

def frame_process(frame, results, model):
    # Draw bounding boxes on the original frame
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Extract the coordinates
        confidence = result.conf[0].item()
        class_id = int(result.cls[0].item())  # Get class id as an integer
        label_name = model.names[class_id]  # Map class id to label name

        label = f"Class: {label_name}, Conf: {confidence:.2f}"

        # Draw the bounding box and label
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/admin')
def dashboard():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # Get the uploaded video file
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join('uploads', video_file.filename)
            video_file.save(video_path)

            output_path = os.path.join('uploads', 'output_' + video_file.filename)
            process_video(video_path, output_path)

            return send_file(output_path, as_attachment=True)

    return render_template('upload.html')

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf[0].item()
            class_index = int(result.cls[0].item())  # Get class index

            # Get the class name using the index
            class_name = CLASS_NAMES.get(class_index, "Unknown")  # Default to "Unknown" if index not found

            # Update label with class name instead of index
            label = f"{class_name}: {confidence:.2f}"
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
