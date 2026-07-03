# RBF Neural Network Implementation Plan for Anti-Ambipolar Transistor Power Prediction

## 1. Project Overview
Implementation of a Gaussian-based RBF (Radial Basis Function) neural network for electricity demand forecasting, based on the paper "Asymmetric-Contact ZnON/DNTT Heterojunctions for Tunable Multi-Gaussian Anti-Ambipolar Responses".

## 2. Project Structure
```
RBF_Power_Prediction/
├── data/
│   └── panama_electricity_data.csv (to be downloaded)
├── src/
│   ├── __init__.py
│   ├── duffing_oscillator.py      # Duffing oscillator data generation
│   ├── gaussian_kernel.py         # Gaussian kernel functions
│   ├── rbf_network.py             # RBF network implementation
│   ├── data_loader.py            # Panama data loader & preprocessing
│   ├── trainer.py                # Training & evaluation
│   └── visualizer.py             # Figure 4 reproduction (j, k plots)
├── main.py                       # Main execution script
├── requirements.txt
└── PLAN.md
```

## 3. Implementation Details

### 3.1 Duffing Oscillator Data Generation
- **Equation**: x'' + 0.2x' + x³ = 0.3cos(1.2t)
- **Solver**: 4th-order Runge-Kutta
- **Time step**: Δt = 1 ms
- **Input window**: 10 consecutive samples (10 ms) → predict next state
- **Total kernels**: 300
- **σ diversity**: 1 to 80 distinct values, evenly distributed

### 3.2 Panama Electricity Demand Data
- **Source**: Panama national grid archive (2016-2020)
- **Training**: 2016-2019 data
- **Testing**: 2020 data (hold-out)
- **Input window**: 10 consecutive days → predict next day
- **Total kernels**: 200
- **σ selection**: Logarithmically spaced within range from Figure 3d-g

### 3.3 Gaussian Kernel Function
```python
def gaussian_kernel(x, center, sigma):
    """
    x: input vector (10-dimensional)
    center: kernel center (10-dimensional)
    sigma: standard deviation (width)
    Returns: Gaussian activation value
    """
    distance = np.linalg.norm(x - center)
    return A * np.exp(-(distance**2) / (2 * sigma**2))
```

### 3.4 RBF Network Architecture
- **Input layer**: 10 nodes (sliding window size)
- **Hidden layer**: N Gaussian kernels (N = 1, 3, 10, 80 for comparison)
- **Output layer**: 1 node (prediction)
- **Activation**: Euclidean distance-based Gaussian
- **Output weights**: Linear least-squares optimization

### 3.5 Training Procedure
1. Cluster input data to find kernel centers (k-means or similar)
2. Calculate Gaussian activations for each input
3. Solve for output weights using least-squares: W = (X^T X)^(-1) X^T y
4. Evaluate on test set

### 3.6 Evaluation Metrics
- **MSE** (Mean Squared Error)
- **R²** (Coefficient of Determination)

## 4. Key Parameters from Paper
| Task | Kernels | σ Range | Window Size | Output |
|------|--------|---------|-------------|--------|
| Duffing | 300 | 1~80 distinct | 10 (10ms) | Next state |
| Panama | 200 | From Fig 3d-g | 10 (days) | Next day |

## 5. Expected Output (Figure 4j, 4k Reproduction)
- MSE vs N (kernel count) plot
- R² vs N (kernel count) plot
- Compare with paper values:
  - N=1: MSE ≈ 9.8×10³ MW², R² ≈ 0.41
  - N≈80: MSE ≈ 1.2×10² MW², R² ≈ 0.94

## 6. Assumptions & Potential Issues
1. **σ range**: Exact σ values from Figure 3d-g not extracted; will estimate based on transfer curve characteristics
2. **Panama data**: Need to obtain from official source or simulate
3. **Clustering method**: Paper mentions "clustering" but doesn't specify algorithm
4. **Weight optimization**: Linear least-squares as specified in paper

## 7. Verification Steps
1. Run with N=1 and verify MSE/R² matches paper (9.8×10³, 0.41)
2. Run with N=80 and verify MSE/R² matches paper (1.2×10², 0.94)
3. Reproduce Figure 4j and 4k curves