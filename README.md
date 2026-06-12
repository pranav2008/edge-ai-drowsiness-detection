# Edge AI Driver Drowsiness Detection System

[![Hardware: Jetson Orin Nano](https://img.shields.io/badge/Hardware-Jetson%20Orin%20Nano-green.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![Framework: OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue.svg)](https://opencv.org/)

A real-time, edge-computing computer vision application designed to monitor driver alertness and mitigate automotive accidents caused by fatigue. Optimized specifically for the **NVIDIA Jetson Orin Nano**, this system processes live video feeds locally to minimize latency, eliminate dependency on cloud connectivity, and ensure driver privacy.

---

## 🚀 Key Features

* **Edge-Optimized Inference:** Built to leverage the NVIDIA Jetson Orin Nano's architecture for low-latency, real-time video processing.
* **Facial Landmark Tracking:** Utilizes computer vision algorithms to map critical facial regions (eyes and mouth) from a live camera feed.
* **Eye Aspect Ratio (EAR) Calculation:** Mathematically monitors eye closure durations to accurately differentiate between normal blinking and micro-sleeps.
* **Yawn Detection (MAR):** Tracks Mouth Aspect Ratio to detect early signs of driver fatigue and deep yawning.

---

## 🛠️ System Architecture & Tech Stack

* **Hardware:** NVIDIA Jetson Orin Nano Developer Kit, USB or CSI Camera Module.
* **Languages:** Python 3.8+
* **Libraries & Frameworks:** OpenCV, NumPy, Dlib / MediaPipe *(adjust based on your exact implementation)*.

---

## 📈 How It Works (The Math)

The system computes the **Eye Aspect Ratio (EAR)** using the Euclidean distance between vertical eye landmarks divided by the distance between horizontal landmarks:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2||p_1 - p_4||}$$

If the EAR falls below a calibrated threshold for a designated number of consecutive frames, a drowsiness trigger is activated. A similar calculation is applied to the **Mouth Aspect Ratio (MAR)** to track fatigue via yawning.

---

## 📦 Installation & Setup

### Prerequisites
* NVIDIA Jetson Orin Nano running JetPack 5.x or higher
* Connected USB or CSI camera module
* Python 3.8+ with `pip` installed

### Steps

1. **Clone the Repository:**
bash
   git clone [https://github.com/pranav2008/edge-ai-drowsiness-detection.git](https://github.com/pranav2008/edge-ai-drowsiness-detection.git)
   cd edge-ai-drowsiness-detection 
bash
   git clone [https://github.com/pranav2008/edge-ai-drowsiness-detection.git](https://github.com/pranav2008/edge-ai-drowsiness-detection.git)
   cd edge-ai-drowsiness-detection
2. **Install Dependencies:**
bash
   pip install -r requirements.txt
3. **Run the Application:**
bash
   python3 interface*
📝 Future Roadmap
   [ ] Optimize the model using NVIDIA TensorRT for higher FP16/INT8 inference frame rates.
   [ ] Implement a physical hardware alert (buzzer/LED) via the Jetson GPIO pins.
   [ ] Add Infrared (IR) camera support for night-time and low-light driving conditions.
