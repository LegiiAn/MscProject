# Vision-Language-Action (VLA) for Transparent Plastic Waste Sorting

This repository contains the code and research for my Master's dissertation in Robotics at Heriot-Watt University.

## Project Overview
Standard robot depth sensors fail to detect transparent plastics, and no training data exists for this specific task. This project tests whether a 7-billion-parameter Vision-Language-Action (VLA) model can learn to sort transparent waste using entirely synthetic training data.

## Architecture
* **Data Generation (ReSort-IT):** A custom tool that builds synthetic training datasets containing images, text instructions, and robot actions.
* **Perception Pipeline:** Uses a ResNet-50 network to predict surface normals from standard RGB images, reconstructing missing depth data without relying on infrared sensors.
* **Control Policy:** The OpenVLA 7B model, fine-tuned using LoRA to map visual and text inputs directly to robotic movements.

## Repository Structure
* `src/`: Code for the ReSort-IT generator, perception pipeline, and VLA control scripts.
* `H00512699_MScProject_Dissertation_MeaVittot.pdf`: Full academic dissertation, ablation studies, and methodology.
* `H00512699_MScProject_Presentation_MeaVittot.pdf`: Presentation slides summarizing the core findings.

## Key Findings
* **Perception Success:** The depth reconstruction system worked effectively, achieving 91.1% segmentation accuracy and adapting well to new shapes.
* **Grasping Performance:** After debugging data pipeline defects, the exact on-target grasp rate reached 7.2%.
* **The Synthetic Data Gap:** The model performed significantly better on real backgrounds (11.3%) compared to synthetic composited ones (3.1%).
* **Conclusion:** Training VLAs on transparent objects using only synthetic data is insufficient. While the model learned text instructions perfectly, it failed to understand physical space. It relies on real-world physical cues, such as light refraction, that synthetic data cannot accurately replicate.

## Author
**Méa Vittot**
Robotics & Artificial Intelligence Engineer
[LinkedIn](YOUR_LINKEDIN_URL)