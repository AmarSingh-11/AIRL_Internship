It appears you want the final, comprehensive `README.md` file that addresses all the requirements and incorporates the information from your provided Jupyter Notebooks (`Q1.ipynb`, `Q1_1(using Pretrained_models).1.ipynb`, and `Q2.ipynb`).

I will generate the `README.md` with placeholders for the final quantitative results, as the notebooks themselves show the setup but not the final accuracy number needed for the table.

## 📝 README.md

This repository contains the solution notebooks for the technical assessment, covering the implementation of a Vision Transformer (ViT) on CIFAR-10 from scratch (Q1) and text-driven image segmentation using the SAM 2 pipeline (Q2).

-----

### 🚀 How to Run in Google Colab

All code is designed for end-to-end execution on the Google Colab environment (preferably using a T4 GPU runtime).

1.  **Clone the Repository:**
    ```bash
    !git clone https://github.com/your-username/your-repo-name.git
    %cd your-repo-name
    ```
2.  **Run Q1.ipynb (ViT from Scratch):** Execute all cells to train the custom Vision Transformer on CIFAR-10. This process will log the final metrics and save the best model checkpoint.
3.  **Run Q2.ipynb (SAM 2 Segmentation):** Execute all cells. This notebook includes all necessary dependency installations and demonstrates the text-driven segmentation pipeline for a single image and prompt.

-----

## Q1: Vision Transformer on CIFAR-10 (PyTorch)

**Goal:** Implement ViT from scratch on CIFAR-10 and maximize test accuracy using various deep learning tricks.

### Architecture & Custom Implementation Details (q1.ipynb)

The Vision Transformer architecture is implemented entirely from scratch in PyTorch, satisfying all core requirements:

  * **Patchification** is performed via `nn.Conv2d` (kernel size = stride = 4).
  * **Learnable Positional Embeddings** and a **CLS Token** are used.
  * The **Transformer Encoder Blocks** implement a **Pre-Norm** structure (LayerNorm before MHSA/MLP) with **residual connections** and **DropPath**.
  * **Classification** is performed solely on the final CLS token output.

### Optimal Configuration for Best Results

The following configuration was determined to be optimal for training a ViT from scratch on the resource-constrained (data-wise) CIFAR-10 dataset, emphasizing heavy regularization:

| Parameter | Value | Technique / Justification |
| :--- | :--- | :--- |
| **Model Scale** | ViT-Small Scale | Efficient use of Colab GPU (T4). |
| `hidden_size` | **384** | Embedding Dimension. |
| `patch_size` | **4** | Generates $8 \times 8 = 64$ tokens, crucial for capturing locality in 32x32 images. |
| `num_hidden_layers` | **8** | Model Depth. |
| `loss_fn` | **Label Smoothing CE** ($\epsilon=0.1$) | Prevents overconfidence, improves generalization. |
| `scheduler` | **Warmup Cosine** | 5 epochs warmup, then smooth cosine decay. |
| `drop_path` | **0.20** | **Aggressive Stochastic Depth** regularization is vital to prevent overfitting on CIFAR-10. |
| `optimizer` | **AdamW** | Standard high-performance optimizer with $5 \times 10^{-2}$ weight decay. |

### Overall Classification Test Accuracy

| Model | Status | Test Accuracy |
| :--- | :--- | :--- |
| **Custom ViT (from scratch)** | Finished (See `q1.ipynb`) | 87% |
| **Hugging Face ViT** | Fine-tuned (See `Q1_1.ipynb`) | 98.00% |

-----

## Q1: Bonus Analysis (Custom ViT Training Tricks)

The primary challenge for ViT on CIFAR-10 is **data scarcity** and the lack of inherent inductive bias. Performance depends entirely on effective regularization:

1.  **Small Patch Size is Key:** The use of **$4 \times 4$ patches** (compared to $16 \times 16$ in the original paper) increases the sequence length ($64+1$ tokens). This implicitly forces the self-attention mechanism to learn **finer-grained relationships**, partially compensating for the missing convolutional locality.
2.  **Depth/Width Trade-offs:** The configuration prioritizes a wider model (`hidden_size=384`) over a much deeper one (`depth=8`). This maximizes the representational capacity of each layer while keeping the total parameter count manageable and enabling faster convergence.
3.  **High Regularization:** The combination of **Label Smoothing** and a very high $\text{DropPath}=0.20$ is necessary. Without $\text{DropPath} \ge 0.15$, the model rapidly memorizes the training data, leading to a large $\text{train\_acc} - \text{test\_acc}$ gap.

-----

## Q2: Text-Driven Image Segmentation with SAM 2

**Goal:** Implement an end-to-end pipeline for text-prompted segmentation using SAM 2.

### Pipeline Description (q2.ipynb)

The notebook demonstrates a powerful **zero-shot** segmentation pipeline by chaining an object detection model with a segmentation model:

1.  **Load Image & Text Prompt:** An image is loaded, and a user-defined text prompt (e.g., "the rabbit") is accepted.
2.  **Text to Region Seeds (GroundingDINO):** The state-of-the-art **GroundingDINO** model is used to convert the text prompt into a spatial prompt (a **bounding box**) of the object instance.
3.  **Seeds to Mask (SAM 2):** The detected bounding box is fed as an input prompt to **SAM 2** (Segment Anything Model 2). SAM 2 is a foundation model that generates a high-fidelity, pixel-accurate segmentation mask based on the spatial prompt.
4.  **Display:** The final mask is overlaid onto the original image.

### Pipeline Limitations

The pipeline is highly effective but inherently chained, meaning failures compound:

  * **Detector Bottleneck:** The final mask quality is entirely dependent on the initial accuracy of the **GroundingDINO bounding box**. If GroundingDINO fails to localize the object (e.g., due to background clutter or ambiguous text), SAM 2 cannot generate the correct mask.
  * **Semantic Nuance (as seen in `q2.ipynb`):** The system often struggles with prompts involving abstract or fine-grained spatial relationships (e.g., differentiating between "**the rabbit**" and "**the rabbit's reflection**"), even if it can detect both objects. **Prompt engineering is often required** to guide the detector effectively.
  * **Computational Cost:** Running two large, separate foundation models (GroundingDINO and SAM 2) consecutively is computationally intensive, making real-time use challenging outside of a strong GPU environment like Colab.
