# 😴 Driver Drowsiness Detection

## 📌 Project Overview
Driver Drowsiness Detection is a real-time computer vision project that uses Python, OpenCV, and MediaPipe to monitor a driver's eyes through a webcam and identify signs of drowsiness. When prolonged eye closure is detected, the system plays an audio alert to warn the driver, helping to improve road safety.

## ✨ Features
* Real-time drowsiness detection using a webcam.
* Facial landmark detection using MediaPipe.
* Eye movement monitoring using the Eye Aspect Ratio (EAR).
* Audio alert when prolonged eye closure is detected.
* Fast and lightweight, runs on a normal laptop.

## 🛠️ Technologies Used
* Python
* OpenCV
* MediaPipe
* NumPy
* Computer Vision

## 👁️ Eye Aspect Ratio (EAR)
The Eye Aspect Ratio is calculated from the facial landmarks around the eye:

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 x ||p1 - p4||)

* When the eyes are open, the EAR stays at a normal value.
* When the eyes close, the EAR drops below a threshold.
* If the EAR stays low for several continuous frames, the driver is considered drowsy.

## 🚀 How It Works
1. Access the webcam and capture live video frames.
2. Detect the face and facial landmarks using MediaPipe.
3. Extract the eye landmarks and calculate the EAR.
4. Compare the EAR with the drowsiness threshold.
5. Play the audio alert if the eyes stay closed for too long.

## ⚙️ Installation and Usage
1. Clone the repository:
   git clone https://github.com/salima45/Driver-Drowsiness-Detection.git
2. Install the required libraries:
   pip install opencv-python mediapipe numpy
3. Run the project:
   python your_file_name.py
4. Press **q** to quit.

## 🎯 Applications
* Driver Safety Systems
* Road Accident Prevention
* Fleet and Transport Monitoring
* Smart Vehicles
* Educational AI Projects

## 🔮 Future Enhancements
* Yawning detection.
* Head pose and distraction detection.
* Mobile application support.
* Integration with in-car IoT systems.

## 👩‍💻 Author
Salima Be
B.Tech – Artificial Intelligence and Data Science
GitHub: https://github.com/salima45
