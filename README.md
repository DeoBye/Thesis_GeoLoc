# Thesis_GeoLoc
**Boyi's Master Thesis -- Geolocalization**

This repository serves as a work log for Boyi's master thesis on geolocalization. 
---

## To-Do List

### 10.15 - 10.24

- [x] **Paper Review**: [*Hard Negative Sampling For Cross-View Geo-Localization*](https://arxiv.org/abs/2303.11851)
- [x] **Code Review**: Explored repository for **SampleGeo**
- [x] **Paper Review**: [*Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization*](https://arxiv.org/pdf/2309.16020v2)
- [x] **Code Review**: Explored repository for **GeoClip**
- [x] **Dataset Handling**: Familiarized with the **MP-16 Dataset** used in GeoClip
- [x] **Script Reproduction**: Reproduced the training script for **GeoClip**
- [ ] **Training Task**: Train **SampleGeo** to assess performance
- [x] **Training Task**: Train **GeoClip** to assess performance

---

## Recent Work Update  11.5 - 11.19

### Dataset
- Downloaded partial [**NUS Global Streetscapes**](https://ual.sg/project/global-streetscapes/) dataset (~25,000 images).

### Training
- Successfully ran the training process using the provided code.
- Noted that the original code did not include the step for adding noise to the location embeddings, as described in the paper. Implemented this step manually.

### Findings
- Trained the model for 20 epochs using a batch size of 64:
  - Adding noise to the embeddings resulted in a lower final loss compared to training without noise.
  - seems convergence was faster with noise addition.
- Evaluated performance based on the distance between predictions and labels at 1, 25, 200, 750, and 2500 km.
- Compared results between two models:
  - **Model 1**: Trained on our data.
  - **Model 2**: Pretrained model provided by the author, validated on the same dataset.
- **Observations**:
  - Our trained model performed worse at the 1 km threshold but achieved better accuracy at all other distances.

### Challenges
- Seems the loss function could potentially be further optimized.
- Observed that while loss decreases, performance improves across all distance thresholds—except for the 1 km threshold, where accuracy declines. 

---

