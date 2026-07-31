# PROJECT HANDOFF — MSc Dissertation: VLA for Transparent Waste Sorting
# Paste this ENTIRE document as your first message in the new conversation.
# Last updated: July 2025

## Student
Méa Vittot, MSc Robotics, Heriot-Watt University (EECE)
Supervisor: Dr. Maria Koskinopoulou
Presentation: August 3-4, 2025
Submission: August 12, 2025
Word limit: 12,000-15,000

## Project Title
Vision-Language-Action (VLA) Models for Semantic Plastic Waste Sorting with Transparent Object Perception

## Architecture
Cascaded 3-stage pipeline:
- WP1 (ReSort-IT): Procedural synthetic data generator — composites transparent ClearGrasp objects onto backgrounds, derives per-image grasp targets + 7-token action strings for OpenVLA
- WP2 (Perception): Dual-head DeepLabV3/ResNet-50 predicting surface normals + segmentation masks from RGB, feeding a Poisson depth solver
- WP3 (VLA Policy): OpenVLA-7B fine-tuned with LoRA (r=32, α=32, 0.22% trainable params) on WP1 data, 4-bit NF4 quantization, single A100

## Cluster
University compute: hwls001 via jump host, single A100, tmux sessions
Python 3.12, PyTorch, HuggingFace Transformers+PEFT, bitsandbytes
Environment: vla_sorting_env
Working dir: ~/MscProject/

## Dataset
Source: ClearGrasp dataset (square-plastic-bottle-train category: 11,702 images with RGB, masks, normals, depth EXR files)
Background pool: 153 texture images in ~/MscProject/data/raw_backgrounds/
Only ONE category on the cluster. Other 4 categories exist locally on Windows only.

## What Has Been Completed

### Run 1 (archived as processed_dataset_run1 and vla_checkpoints_run1)
- Generated 24,570 train / 5,265 val / 5,268 test composites at 640x480
- Augmentations: scale 0.25-0.75, flip, rotation ±15°, photometric jitter, hue/sat, Gaussian blur on mask edges (seam removal), Gaussian noise, background-identity split between train/val/test
- VLA trained: lr=1e-5, batch=2, patience=15, val every 500 steps, early stop at step 21,500, best val loss 0.154
- Best checkpoint: lora_epoch_2_step_13000

### Run 1 Measured Results
**Perception (WP2):**
- In-distribution (square-plastic-bottle, 11,701 images): Mean angular error 19.55°, median 14.29°, RMSE 25.57°, <11.25°: 39.3%, <22.5°: 69.4%, IoU 91.10%
- OOD (cup-with-waves, 100 imgs): Mean 34.06°, IoU 77.28%
- OOD (champagne-glass, 117 imgs): Mean 25.53°, IoU 77.87%

**Depth Ablation (100 images, 0.5x res, fx=fy=459):**
- Raw corrupted: 0.9297m RMSE
- cv2.inpaint: 0.1227m RMSE [0.1152, 0.1306] ← BEST
- Poisson + GT normals: 0.1364m [0.1250, 0.1490]
- Poisson + predicted normals: 0.4160m [0.3936, 0.4390]
- Wilcoxon: inpaint vs Poisson-GT p=0.02 (inpaint wins); inpaint vs Poisson-pred p=3.9e-18
- KEY FINDING: Inpainting is SUPERIOR to the Poisson solver even with perfect normals, for these shallow objects. Solver is unjustified for this morphology.

**Lambda sensitivity (30 images):** Stable at λ=10,50,100 (~0.126-0.128m); degrades at 500 (0.214m), 1000 (0.407m)

**VLA Spatial Accuracy:**
- Best checkpoint (step 13000): Mean L1 27.6 bins, median 26.5, X=32.7, Y=32.5
- Error distribution: unimodal, 10-46 bins, no bimodal hallucination mode
- Centre-prior comparison: a static (112,112) predictor would score ~33/30 bins per axis — model's 32.7/32.5 is INDISTINGUISHABLE from this baseline in aggregate
- Evidence of image-conditioning exists at sample level (predictions shift toward targets) but mean precision does not beat the prior

**Grasp Success (exact mask measurement, 190 samples):**
- ON-TARGET: 2.6% [1.1%, 6.0%]
- ON-OBJECT: 4.7% [2.5%, 8.8%]
- PROXIMITY (40px): 14.7% [10.4%, 20.5%]
- GT label validity: only 90.0% (171/190) — 10% of training labels were invalid (centroid outside concave objects)
- Mean pred-GT distance: 114px

**Latency (A100, 20 runs):**
- Perception: 15.4ms (0.4%)
- Depth reconstruction: 2513ms (65.0%)
- VLA generation: 1337ms (34.6%)
- Total: 3865ms = 0.26 FPS

**Coordinate distribution:** Near-centred (X mean=318 vs centre=320, Y mean=253 vs centre=240), bell-shaped (σ_x=104, σ_y=93), non-uniform (0.80 X, 0.62 Y)

### Critical Bugs Found and Fixed During Project
1. VLA dataset loader was feeding DUMMY CONSTANT actions (not ReSort-IT targets) → loss hit 0.0000, 97.7% token accuracy was meaningless, model memorized one coordinate pair
2. BGR→XYZ channel swap needed for ClearGrasp EXR normals loaded via OpenCV
3. Perspective focal-length scaling (÷fx) needed in Poisson gradient constraints — without it, gradients were 460x too large
4. Gradient constraints must be restricted to hole-region only + strong anchor weighting (λ=100)
5. 10% label poisoning from centroid-outside-concave-objects

## Run 2 (IN PROGRESS — not yet started on cluster)
Three root-cause fixes:
- FIX 1: Validity-guaranteed targets (pointPolygonTest + distance-transform fallback) → 100% label validity
- FIX 2: Native 224x224 resolution (removes double downsampling)
- FIX 3: 50/50 mix of composites + real ClearGrasp native frames (restoring refraction/shadow/specular cues)

Files ready to push:
- generatedata.py v4 (with all 3 fixes + mask saving)
- verify_labels.py (pre-training audit, must print PASS)
- grasp_success_proxy.py v3 (3 criteria + per-type + per-size breakdowns)

Step-by-step for Run 2:
1. Archive Run 1: mv processed_dataset processed_dataset_run1; mv vla_checkpoints vla_checkpoints_run1
2. Generate new dataset (~25-35 min CPU)
3. verify_labels.py → must print PASS
4. Retrain VLA (tune_vla.py unchanged, overnight)
5. evaluate_checkpoints.py + plot_checkpoint_curves.py
6. grasp_success_proxy.py v3

## Generated PDFs/Figures (all on cluster + local)
- perception_evaluation_full.pdf (ID only — histogram, CDF, qualitative example)
- ablation_depth_reconstruction.pdf (3 qualitative rows + boxplot + bar chart)
- ablation_statistics.pdf (bootstrap CIs)
- lambda_sensitivity.pdf (RMSE vs λ, log scale)
- coordinate_distribution.pdf (X/Y histograms, 2D heatmap, scatter)
- vla_error_distribution.pdf (L1 histogram, CDF, per-axis scatter)
- grasp_success_gallery.pdf (3 hits + 3 misses, proximity criterion)
- pipeline_gallery.pdf (4 ClearGrasp scenes: RGB→normals→depth→grasp overlay)
- latency_benchmark.json

## Dissertation State
LaTeX in ~/MscProject/dissertation/chapters/
- chapter1_introduction.tex — WRITTEN (v1, needs RQ5 placeholder filled)
- chapter2_litreview.tex — WRITTEN (v2, merged from student's original + corrections, ~2600 words, expand toward 4000)
- chapter3_methodology.tex — WRITTEN (v1, needs patches M1+M2 from run2)
- chapter4_results.tex — WRITTEN (v2, needs patches R1-R4 from run2 + Run 2 TBD slots filled)
- chapter5_discussion.tex — WRITTEN (v1, needs centre-prior harshening + run2 discussion)
- chapter6_conclusion.tex — WRITTEN (v1, needs run2 outcomes)
- abstract.tex — WRITTEN (needs "41%" claim removed, replaced with honest framing)
- references.bib — has duplicates to fix (sajjan2020 x2, zitkovich2023 x2)

Key bib entries to add: hu2021lora, chen2017rethinking, dettmers2023qlora
Replace sapkota2025 with li2025survey

## Supervisor Feedback (most recent)
Maria suggested two directions:
1. Semantic reasoning: reasoning about material type, recyclable vs non-recyclable, contamination, language-conditioned instructions ("pick only plastic"), open-world recognition
2. Gazebo simulation: ABB IR360 delta robot, connect VLA output to simulated picks

My recommendation: Option A+D (zero-shot recycling categorization + open-world recognition test on the existing fine-tuned model) as a small 2-3 day experiment AFTER Run 2 results are in. Gazebo only if time permits (3-4 day timebox with Christopher's help). Both secondary to Run 2 and writing.

## Key Architectural Decisions to Defend in Viva
- Single category training: deliberate scoping, OOD evaluation on other categories
- Composites-only → composites+real mix: diagnosed as root cause, Run 2 tests it
- Inpainting vs Poisson: honestly report inpainting wins for shallow objects
- LoRA not full fine-tune: VRAM constraint + catastrophic forgetting prevention
- 256-bin tokenization: OpenVLA convention, resolution studied
- The "memorization bug" story: shows independent thinking and diagnostic rigor — examiners reward this

## What Gemini (the other AI assistant) Contributed
- Correctly identified that Poisson blending would be a debugging rabbit hole → recommended Gaussian mask-blur instead (adopted)
- Correctly identified that reconstruct_depth.py exists and has a Poisson solver (before I had the code)
- Provided educational LoRA explanation useful for methodology chapter
- Some of its framing was overenthusiastic and had to be corrected

## Current Immediate Priority
1. Push Run 2 files to cluster and execute the step-by-step
2. While training runs: apply dissertation patches, expand lit review
3. When Run 2 completes: fill TBD slots, write the interpretive paragraph
4. Then: semantic reasoning experiment if results warrant it, or write the "failure analysis" narrative if they don't
