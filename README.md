# ✨ Air Drawing using Hand Gestures
A Computer Vision project built using **Python**, **MediaPipe**, **OpenCV**, **NumPy**, and **other required modules** that allows users to draw in the air using hand gestures.<br>
The project tracks hand movements through a webcam and converts finger motion into virtual drawing on the screen.

# 💫 Features
- **✋ Real-time hand tracking**
- **🎨 Draw on screen using finger gestures**
- **📷 Webcam-based interaction**
- **⚡ Fast and smooth tracking using MediaPipe**
- **🧠 Gesture detection module**
- **🖌️ Virtual air drawing system**

# 🛠️ Technologies Used
| Technology | Purpose                             |
|------------|-------------------------------------|
| Python     | Main programming language           |
| OpenCV     | Video capturing & image processing  |
| Mediapipe  | Hand landmark detection & tracking  |
| NumPy      | Numerical operations                |
- **Other Modules (Tesseract, Pillow)**

# 📂 Project Structure
**CV_Project/** <br>
│           <br>
├── vision_assets/           **(Assets used in the project)* <br>
├── AeroDrawModule.py        **(Main drawing module)* <br>
├── GestureTracker.py        **(Hand tracking & gesture detection)* <br>
├── hand_landmarker.task     **(MediaPipe hand landmark model)* <br>
├── requirements.txt         **(Required Python modules)* <br>

# Create Virtual Environment (A Better Option)
- Creating a virtual environment keeps project dependencies separate from your system Python. And, <br>
  **📥 Install Required Modules in virtual**
   ```bash
   pip install -r requirements.txt
   ```
- ✍️**NOTE: Download latest version of all required modules in virtual environment**

# 🎮 Controls & Hand Gestures

| Gesture / Key | Function |
|---|---|
| ☝ Index Finger | Draw letter in air |
| ✌ Index + Middle Finger | Select colour from top bar |
| 🖐 All 5 Fingers | Recognize letter and add to word |
| ⌨️ SPACE Key | Add space between words |
| ⌫ BACKSPACE Key | Delete last letter |
| 🔤 A Key | Clear all words |
| ❌ Q Key | Quit the application |


# 📸 How It Works
**1. Webcam captures hand movements.** <br>
**2. MediaPipe detects hand landmarks.** <br>
**3. Finger positions are tracked in real time.** <br>
**4. Gestures are interpreted as drawing actions.** <br>
**5. Virtual drawing appears on the screen.** <br>

# 👨‍💻 Author
- **Sahil Verma**
  
