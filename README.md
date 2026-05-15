# NL2Net Reproduction

This repository reproduces **NL2Net: Non-Local and Local Feature-Coupled Self-Supervised Network for Hyperspectral Anomaly Detection**.

The code is based on the official NL2Net implementation and has been modified for easier reproduction. The current version supports running training and prediction in one command, reads datasets from the local `./data` folder, and automatically outputs an evaluation summary table.

---

## 1. Task

This project focuses on **hyperspectral anomaly detection (HAD)**.  
The model reconstructs the hyperspectral background and uses the reconstruction difference to generate an anomaly detection map. The final evaluation metric is **AUC**.

---

## 2. Framework Used

- Python
- PyTorch
- Self-supervised hyperspectral anomaly detection
- NL2Net dual-branch network
- Local feature extraction + non-local self-attention feature extraction

---

## 3. Environment

Recommended environment:

```text
Python 3.9
PyTorch 1.12.1
NumPy 1.21.5
SciPy 1.7.3
TorchVision 0.13.1
scikit-learn
TensorBoard
einops
```

Install dependencies:

```bash
pip install torch torchvision numpy scipy scikit-learn tensorboard einops
```

If using Anaconda:

```bash
conda create -n nl2net python=3.9
conda activate nl2net
pip install torch torchvision numpy scipy scikit-learn tensorboard einops
```

---

## 4. Project Structure

Please keep the files in the following structure:

```text
Repro1/
├── main.py
├── dataset.py
├── model.py
├── utils.py
├── data/
│   └── HSI-II.mat
├── checkpoints/
│   └── ...
└── result/
    └── ...
```

The code reads the dataset from the `data` folder under the same directory as `main.py` and `dataset.py`.

---

## 5. Dataset Format

The dataset should be stored as a `.mat` file under:

```text
./data/HSI-II.mat
```

The `.mat` file must contain:

```text
data : hyperspectral image cube, shape = H × W × Bands
map  : binary ground-truth anomaly map, shape = H × W
```

For the reproduced experiment, `HSI-II.mat` contains:

```text
data shape = 150 × 150 × 102
map shape  = 150 × 150
map values = 0 / 1
```

This corresponds to the **Pavia anomaly detection dataset** used in the NL2Net paper.

---

## 6. Run the Code

### Train only

```bash
python main.py --command train --dataset HSI-II --epochs 5000 --learning_rate 1e-4 --factor 3 --gpu_ids 0
```

### Predict only

```bash
python main.py --command predict --dataset HSI-II --epochs 5000 --learning_rate 1e-4 --factor 3 --gpu_ids 0
```

### Train and predict together

```bash
python main.py --command train_predict --dataset HSI-II --epochs 5000 --learning_rate 1e-4 --factor 3 --gpu_ids 0
```

The default command in this modified version is `train_predict`, so pressing **F5** in VS Code can directly run training followed by prediction if the launch configuration uses `main.py`.

---

## 7. Output Files

After running prediction, the outputs are saved under:

```text
./result/HSI-II/
```

Main output files:

```text
detectmap.mat              anomaly detection map
reconstructed_data.mat     reconstructed hyperspectral data
evaluation_summary.csv     evaluation result table
evaluation_summary.txt     readable evaluation summary
log.txt                    prediction log
```

Model checkpoints are saved under:

```text
./checkpoints/HSI-II/
```

---

## 8. Evaluation Metric

The main metric is **AUC**.

The model computes the anomaly score map from the reconstruction difference between the original hyperspectral image and the reconstructed image. The AUC is then calculated using the binary ground-truth anomaly map.

---

## 9. Reproduced Result

Using the following setting:

```text
dataset       = HSI-II / Pavia
epochs        = 5000
learning_rate = 1e-4
factor        = 3
gpu_ids       = 0
metric        = AUC
```

The reproduced result is:

```text
AUC = 0.9897690799
Testing time = 0.0584 s
```

Compared with the paper-reported Pavia result:

```text
Paper Pavia AUC = 0.9916
Our AUC         = 0.989769
AUC gap         = -0.001831
```

The reproduced result is close to the paper-reported result. The small gap may be caused by random initialization, CUDA/PyTorch version differences, numerical precision, or minor preprocessing and implementation differences.

---

## 10. Notes

- `dataset.py` has been modified to read data from the current project directory: `./data/`.
- `main.py` has been modified to support `train_predict` and automatic evaluation table output.
- The dataset must contain both `data` and `map`; otherwise AUC cannot be calculated.
- If only the hyperspectral cube is available without a ground-truth map, the model can generate a detection map, but AUC cannot be evaluated.

---

## 11. Citation

If this code or paper is useful, please cite:

```bibtex
@article{wang2025nl2net,
  author={Wang, Degang and Ren, Longfei and Sun, Xu and Gao, Lianru and Chanussot, Jocelyn},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  title={Non-Local and Local Feature-Coupled Self-Supervised Network for Hyperspectral Anomaly Detection},
  year={2025},
  doi={10.1109/JSTARS.2025.3542457}
}
```
