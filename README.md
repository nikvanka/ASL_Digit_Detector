Make sure the following are installed:

1. Python (preferably 3.8 - 3.10)
👉 Download Python

2. pip (comes with Python)
3. Virtual Environment (optional but recommended)
bash
Copy
Edit
python -m venv env
.\env\Scripts\activate       # On Windows
📁 Step-by-Step Setup Guide
🔹 Step 1: Move into the project folder
Go to the folder where your project exists:

bash
Copy
Edit
cd C:\Users\vanka\Desktop\ASL_Digit_Detector
🔹 Step 2: Install the required packages
If you already have a requirements.txt, run:

bash
Copy
Edit
pip install -r requirements.txt
If not, here’s the list you’ll likely need (based on what I saw in the project):

bash
Copy
Edit
pip install opencv-python
pip install mediapipe
pip install numpy
🔹 Step 3: Run the main Python file
Assuming the file is air_whiteboard.py (which handles the webcam input and detection):

bash
Copy
Edit
python air_whiteboard.py
📹 What Happens When You Run It?
Your webcam will turn on.

Your hand is detected using MediaPipe Hands.

Based on finger gestures, it lets you draw on the screen like a whiteboard.

🧠 Where AI is Used?
AI is used in:

Hand tracking & landmark detection (via Google’s MediaPipe which uses deep learning).

Gesture recognition: AI detects your finger position to know if you're pointing, drawing, or selecting colors.

🎨 Features
Detects fingers and draws virtually.

Switch between tools using finger combinations.

Track movement and update drawings in real time.

🛑 Common Errors to Avoid
❌ Error	💡 Fix
cv2 module not found	Run pip install opencv-python
mediapipe module not found	Run pip install mediapipe
Webcam not opening	Make sure no other app is using it
App freezes/crashes	Check Python version (3.8–3.10 recommended)

