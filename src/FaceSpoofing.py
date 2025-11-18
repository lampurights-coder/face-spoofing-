# import cv2
# import dlib
# from src.scrfd import SCRFD
# import numpy as np
# from openvino.runtime import Core
# from collections import deque


# class FaceSpoofDetector:
#     def __init__(self, model_path="./vino/model.xml", device="CPU", model_file='./vino/scrfd_2.5g_bnkps_shape640x640.onnx'):
#         # Initialize OpenVINO model
#         core = Core()
#         model = core.read_model(model_path)
#         self.compiled_model = core.compile_model(model, device)
#         self.detector = SCRFD(model_file=model_file)
#         self.detector.prepare(-1)
#         # Initialize face detector
#         self.face_detector = dlib.get_frontal_face_detector()
        
#         # Normalization parameters
#         self.MEAN = np.array([0.485, 0.456, 0.406]) * 255.0
#         self.STD = np.array([0.229, 0.224, 0.225]) * 255.0
        
#         # To keep track of predictions history per face center
#         self.history_length = 3
#         self.face_histories = {}  # Key: face center tuple (x, y), Value: deque of last labels
    
#     def get_boundingbox(self, face, width, height):
#         """Calculate square bounding box around detected face"""
#         x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
#         size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
#         center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
#         # Calculate boundaries
#         x1 = max(int(center_x - size_bb // 2), 0)
#         y1 = max(int(center_y - size_bb // 2), 0)
#         size_bb = min(width - x1, size_bb, height - y1)
        
#         return x1, y1, size_bb, (center_x, center_y)


#     def preprocess(self, face_roi):
#         """Prepare face ROI for inference"""
#         face_img = cv2.resize(face_roi, (224, 224))
#         face_img = face_img.astype(np.float32)
#         face_img = (face_img - self.MEAN) / self.STD
#         return np.transpose(face_img, (2, 0, 1))[np.newaxis, ...]


#     def softmax(self, x):
#         e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#         return e_x / e_x.sum(axis=-1, keepdims=True)


#     def predict_face(self, face_roi):
#         """Run inference on face region"""
#         input_tensor = self.preprocess(face_roi)
#         output = self.compiled_model([input_tensor])[0]
        
#         probabilities = self.softmax(output)[0]
#         real_prob = float(probabilities[1])
#         fake_prob = float(probabilities[0])
        
#         label = "REAL" if real_prob > 0.7 else "FAKE"
#         # Confidence no longer used for display
#         return label, real_prob, fake_prob


#     def stable_label(self, history):
#         """Return stable label if consistent for last self.history_length frames"""
#         if len(history) < self.history_length:
#             return None  # Not enough data yet
#         # Check if all labels in history are the same
#         if all(lab == history[0] for lab in history):
#             return history[0]
#         return None


#     def process_frame(self, frame):
#         """Process a video frame and draw detection results with stability over frames"""
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = self.face_detector(rgb_frame, 1)
#         print(faces)
#         # Clean old face_centers keys that are not detected in current frame
#         # We'll keep only those close to current detections, simple approximation
#         current_centers = []
#         for face in faces:
#             _, _, _, center = self.get_boundingbox(face, frame.shape[1], frame.shape[0])
#             current_centers.append(center)
#         # Remove face histories for centers not close to any current_center (distance > 50 px)
#         keys_to_remove = []
#         for key in self.face_histories.keys():
#             if all(np.linalg.norm(np.array(key) - np.array(c)) > 50 for c in current_centers):
#                 keys_to_remove.append(key)
#         for key in keys_to_remove:
#             del self.face_histories[key]

#         for face in faces:
#             x, y, size, center = self.get_boundingbox(face, frame.shape[1], frame.shape[0])
#             face_roi = frame[y:y+size, x:x+size]
#             if face_roi.size == 0:
#                 continue
            
#             label, real_prob, fake_prob = self.predict_face(face_roi)
            
#             # Update history
#             if center not in self.face_histories:
#                 self.face_histories[center] = deque(maxlen=self.history_length)
#             self.face_histories[center].append(label)
            
#             stable = self.stable_label(self.face_histories[center])
#             if stable is not None:
#                 color = (0, 255, 0) if stable == "REAL" else (0, 0, 255)
#                 cv2.rectangle(frame, (x, y), (x + size, y + size), color, 2)
#                 # Draw label only when stable, without confidence
#                 cv2.putText(frame, stable,
#                             (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7, color, 2)
#             else:
#                 # Optionally draw grey box while decision not stable
#                 cv2.rectangle(frame, (x, y), (x + size, y + size), (128, 128, 128), 1)
        
#         return frame


#     def process_video(self, video_path='/home/arshia/Downloads/projects/spoof/3.mp4', output_file="output.mp4"):
#         cap = cv2.VideoCapture(video_path)
        
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
        
#         out = None
#         if output_file:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             processed_frame = self.process_frame(frame)
            
#             cv2.imshow('Face Spoof Detection', processed_frame)
#             if out:
#                 out.write(processed_frame)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         cap.release()
#         if out:
#             out.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     detector = FaceSpoofDetector()
#     input_source = '/home/arshia/Downloads/projects/spoof/grabage/IMG_8142.MOV'  # or 0 for webcam
#     detector.process_video(video_path=input_source, output_file="output_2.mp4")
import cv2
import numpy as np
from src.scrfd import SCRFD
from openvino.runtime import Core
from collections import deque


class FaceSpoofDetector:
    def __init__(self, model_path="./vino/model.xml", device="CPU", model_file='./vino/scrfd_2.5g_bnkps_shape640x640.onnx'):
        # Initialize OpenVINO model
        core = Core()
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, device)
        self.detector = SCRFD(model_file=model_file)
        self.detector.prepare(-1)
        
        # Normalization parameters
        self.MEAN = np.array([0.485, 0.456, 0.406]) * 255.0
        self.STD = np.array([0.229, 0.224, 0.225]) * 255.0
        
        # To keep track of predictions history per face center
        self.history_length = 1
        self.face_histories = {}  # Key: face center tuple (x, y), Value: deque of last labels
    
    def get_boundingbox(self, face, width, height):
        """Calculate square bounding box around detected face"""
        # SCRFD returns [x1, y1, x2, y2, score]
        x1, y1, x2, y2 = map(int, face[:4])
        size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Calculate boundaries
        x1 = max(int(center_x - size_bb // 2), 0)
        y1 = max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb, height - y1)
        
        return x1, y1, size_bb, (center_x, center_y)


    def preprocess(self, face_roi):
        """Prepare face ROI for inference"""
        face_img = cv2.resize(face_roi, (224, 224))
        face_img = face_img.astype(np.float32)
        face_img = (face_img - self.MEAN) / self.STD
        return np.transpose(face_img, (2, 0, 1))[np.newaxis, ...]


    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


    def predict_face(self, face_roi):
        """Run inference on face region"""
        input_tensor = self.preprocess(face_roi)
        output = self.compiled_model([input_tensor])[0]
        
        probabilities = self.softmax(output)[0]
        real_prob = float(probabilities[1])
        fake_prob = float(probabilities[0])
        print(fake_prob, real_prob)
        label = "REAL" if real_prob > 0.73 else "FAKE"
        return label, real_prob, fake_prob


    def stable_label(self, history):
        """Return stable label if consistent for last self.history_length frames"""
        if len(history) < self.history_length:
            return None  # Not enough data yet
        # Check if all labels in history are the same
        if all(lab == history[0] for lab in history):
            return history[0]
        return None


    def process_frame(self, frame):
        """Process a video frame and draw detection results with stability over frames"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # SCRFD detection: returns bboxes (x1, y1, x2, y2, score) and kpss (keypoints)
        bboxes, kpss = self.detector.detect(rgb_frame)
        # Clean old face_centers keys that are not detected in current frame
        current_centers = []
        for face in bboxes:
            _, _, _, center = self.get_boundingbox(face, frame.shape[1], frame.shape[0])
            current_centers.append(center)
        # Remove face histories for centers not close to any current_center (distance > 50 px)
        keys_to_remove = []
        for key in self.face_histories.keys():
            if all(np.linalg.norm(np.array(key) - np.array(c)) > 50 for c in current_centers):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.face_histories[key]

        for face in bboxes:
            x, y, size, center = self.get_boundingbox(face, frame.shape[1], frame.shape[0])
            face_roi = frame[y:y+size, x:x+size]
            if face_roi.size == 0:
                continue
            
            label, real_prob, fake_prob = self.predict_face(face_roi)
            
            # Update history
            if center not in self.face_histories:
                self.face_histories[center] = deque(maxlen=self.history_length)
            self.face_histories[center].append(label)
            
            stable = self.stable_label(self.face_histories[center])
            if stable is not None:
                color = (0, 255, 0) if stable == "REAL" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + size, y + size), color, 2)
                # Draw label only when stable, without confidence
                cv2.putText(frame, stable,
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
            else:
                # Optionally draw grey box while decision not stable
                cv2.rectangle(frame, (x, y), (x + size, y + size), (128, 128, 128), 1)
        
        return frame


    def process_video(self, video_path='/home/arshia/Downloads/projects/spoof/3.mp4', output_file="output.mp4"):
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('Face Spoof Detection', processed_frame)
            if out:
                out.write(processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FaceSpoofDetector()
    input_source = '/home/arshia/Downloads/projects/spoof/grabage/IMG_8142.MOV'  # or 0 for webcam
    detector.process_video(video_path=input_source, output_file="output_2.mp4")