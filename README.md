# Vision-Language-Action (VLA) for Transparent Plastic Waste Sorting

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenVLA](https://img.shields.io/badge/Model-OpenVLA_7B-orange)
![Computer Vision](https://img.shields.io/badge/Perception-ResNet50-blue)

This repository contains the code and findings for my Master's dissertation in Robotics at Heriot-Watt University. 

**The Challenge:** Transparent plastics defeat standard active infrared depth sensors used in robotic sorting, and there are no existing datasets of robot demonstrations for handling transparent waste. 
**The Objective:** Investigate whether a 7-billion-parameter Vision-Language-Action (VLA) policy can learn to locate and sort transparent waste using *entirely synthetic training data*.

## 🧠 Core Architecture
To address hardware limitations and data scarcity, this project utilizes a cascaded software framework:

* **Procedural Data Generation (ReSort-IT):** A custom generator that produces synthetic image-language-action training datasets specifically for transparent object manipulation.
* **Geometry-Aware Perception:** Bypasses IR sensor failures by predicting surface normals from RGB images (via ResNet-50) and reconstructing the missing metric depth using a normal-guided Poisson solver.
* **VLA Policy Fine-Tuning:** Parameter-efficient fine-tuning (PEFT) of the 7B-parameter OpenVLA model using LoRA.

## 📂 Repository Structure
```text
📦 MscProject
 ┣ 📂 src/                                       # ReSort-IT generator, perception pipeline, and VLA control scripts
 ┣ 📜 H00512699_MScProject_Dissertation_MeaVittot.pdf # Full methodology, mathematical proofs, and ablation studies
 ┣ 📜 H00512699_MScProject_Presentation_MeaVittot.pdf # Slide deck summarizing core findings
 ┗ 📜 README.md