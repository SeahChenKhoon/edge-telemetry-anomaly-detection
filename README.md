# Benchmarking Isolation Forest and ONNX-Accelerated Proxy Models for Anomaly Detection

This project compares the performance of a native `IsolationForest` model with an ONNX-accelerated proxy model for anomaly detection. Since `IsolationForest` is not natively supported in ONNX, a `RandomForestClassifier` was trained on its anomaly labels to act as a supervised proxy, allowing ONNX export.

All models were evaluated using CPU-only execution, with a focus on average inference time per sample and model file size. The ONNX model—despite being derived from a larger supervised model—achieved the fastest inference and demonstrated competitive model size, making it well-suited for deployment in latency-sensitive or resource-constrained environments.

---
## 🛰️ Project Overview

The **Anomaly Detection Optimization Framework** is a lightweight, deployment-ready pipeline for identifying abnormal patterns in telemetry data with minimal latency and resource overhead. It is purpose-built for **real-time anomaly detection in edge and embedded systems**, where compute and memory resources are limited.

The framework begins with exploratory analysis of telemetry metrics—such as CPU usage, memory consumption, network traffic, and task-level behaviors—drawn from a real-world dataset. An unsupervised `IsolationForest` model is first trained to detect anomalies. These outputs are then used as pseudo-labels to train a supervised `RandomForestClassifier`, which enables conversion to the **ONNX** format for efficient and portable inference.

To validate performance, the system benchmarks model size and average inference time across native and *ONNX* formats. The ONNX model runs **20–25× faster per sample** compared to the original implementation, while maintaining comparable anomaly detection logic. This confirms its suitability for streaming or batch inference in **CPU-only and resource-constrained environments.**

🔧 Key Use Cases:
- Improves flow and clarity with logical sequencing
- Uses consistent verb tense and more formal language
- Emphasizes ONNX advantages clearly (speed, portability)
- Adds sectioning (Key Use Cases) for readability

---

## 🔧 Features
- **Unsupervised anomaly detection** using IsolationForest on telemetry metrics (CPU, memory, power, etc.)
- **Supervised proxy modeling** via RandomForestClassifier trained on pseudo-labels for ONNX compatibility
- **ONNX model export** for platform-agnostic, deployment-ready inference
- **Simulated real-time benchmarking** with per-sample latency profiling on CPU
- **Telemetry dataset integratio**n via Hugging Face (intel-cpu-dataset)
- **Exploratory data analysis (EDA)** and anomaly score visualization to interpret model behavior
- **Model evaluation report** comparing file size and inference speed across native and ONNX formats

---

## 📁 Project Structure
``` text
├── EDA.ipynb                                  # Jupyter notebook containing EDA, model training, and ONNX export
├── intel_cpu_dataset.csv                      # Raw telemetry dataset used for anomaly detection
├── isolation_forest_optimized.pkl             # Trained Isolation Forest model (used to generate pseudo-labels)
│
├── rf_anomaly_detector.onnx                   # ONNX model exported from Random Forest classifier
├── rf_anomaly_detector_quant.onnx             # Quantized version of the ONNX model (int8)
├── rf_test_opset11.onnx                       # Test export for ONNX opset version verification
│
├── README.md                                  # Project overview, setup instructions, and evaluation results
├── .gitignore                                 # Files and directories ignored by Git
├── mlruns/                                    # MLflow experiment tracking directory
```
---
## 🔧 Installation
Follow these steps to set up and run the Edge Telemetry Anomaly Detection project:

1. Clone the Repository
Open your terminal (e.g., Git Bash), navigate to your desired directory, and run:
```
git clone https://github.com/SeahChenKhoon/edge-telemetry-anomaly-detection.git
cd edge-telemetry-anomaly-detection
```
![Clone the Repository](images/01_Clone_the_Repository.jpg)

2. Set Up a Virtual Environment
````text
python -m venv venv
venv\Scripts\activate  
````
![Set Up a Virtual Environment](images/02_Set_Up_a_Virtual_Environment.jpg)

3. Run the Notebook
- Launch Jupyter Notebook or JupyterLab:
- Open EDA.ipynb and run through the cells to:
  - Load telemetry data
  - Perform EDA
  - Train Isolation Forest and Random Forest proxy models
  - Export to ONNX
  - Run real-time inference benchmarks
---
## 📄 Project Report
#### 🧭 Scenario Overview
Modern edge devices and industrial systems increasingly depend on real-time telemetry data to ensure reliability, safety, and performance. These devices generate high-frequency measurements such as CPU usage, memory consumption, execution time, and power usage — all of which can be monitored to detect early signs of system stress, hardware failure, or performance degradation.

The objective of this project is to **develop and evaluate a lightweight anomaly detection pipeline** capable of operating efficiently in **resource-constrained environments**. Specifically, the solution must support:

Real-time inference on edge devices (without GPU acceleration)

Low memory and storage footprint

Accurate detection of abnormal system behavior based on telemetry inputs

To accomplish this, we use a machine learning–based anomaly detection strategy that combines **unsupervised learning (Isolation Forest)** with a **supervised proxy model (Random Forest)** for ONNX compatibility. The goal is to benchmark both models and optimize for deployment via ONNX Runtime.

#### 📊 Chosen Dataset: Intel CPU Telemetry Dataset
- Dataset Name: `intel-cpu-dataset`
- Source: [Intel CPU Telemetry Dataset on Hugging Face](https://huggingface.co/datasets/MounikaV/intel-cpu-dataset)
- Format: CSV (12 columns × 2081 rows)


#### 🧾 Features Included

| Feature Name                  | Description                                        |
|------------------------------|----------------------------------------------------|
| `vm_id`, `timestamp`         | Identifiers for system and time series            |
| `cpu_usage`                  | CPU utilization percentage                         |
| `memory_usage`               | Memory consumption in MB                           |
| `network_traffic`            | Network activity in MB/s                           |
| `power_consumption`          | Power draw in watts                                |
| `num_executed_instructions`  | Instruction count for tasks                        |
| `execution_time`             | Time taken to complete task (ms)                   |
| `energy_efficiency`          | Derived metric: work done per energy unit          |
| `task_priority`              | Integer indicating criticality level               |
| `task_type_*` (3 binary fields) | Task workload type: compute, IO, network       |
| `task_status`                | Label: 0 = normal, 1 = failed (target for detection) |

#### 🧠 Relevance to PC Telemetry
This dataset captures the exact kind of low-level hardware telemetry used in real-world edge monitoring systems. Each row represents a snapshot of system behavior during task execution, and failure labels indicate deviations from normal behavior, making it ideal for training anomaly detection models. The fields align with commonly observed PC metrics like:
- Resource usage (CPU, memory)
- Execution efficiency (power, time)
- Operational context (task type and priority)

These features simulate data you'd collect from an edge device, such as an embedded controller, industrial PC, or IoT system, making this dataset a realistic proxy for PC telemetry applications.


### 🔍 Key Findings from EDA
The exploratory data analysis (EDA) revealed distinct patterns separating normal and anomalous task executions in the telemetry dataset:
| Aspect              | Normal Behavior                              | Anomalous Behavior (Failures)                          |
|---------------------|-----------------------------------------------|--------------------------------------------------------|
| `cpu_usage`         | Generally moderate (20–60%)                  | Often abnormally high (>80%) or unusually low         |
| `memory_usage`      | Consistently within mid-range values         | Sharp spikes or drops near system limits              |
| `network_traffic`   | Normalized traffic depending on task type    | Sudden drops or idle behavior in network tasks        |
| `power_consumption` | Correlated with CPU/memory use               | High power draw without matching performance          |
| `execution_time`    | Relatively consistent per task type          | Often significantly prolonged                         |
| `energy_efficiency` | Stable, high-efficiency ratio                | Marked degradation (lower work-per-watt)              |
| `task_status`       | `0` (Normal)                                 | `1` (Failure / Anomaly)                               |


#### 📈 Distributional Insight:
- Anomaly scores from IsolationForest showed clear right-skewed distribution, with failed samples having higher scores.
- Visualizations indicated that anomalies cluster in regions of extreme feature values (e.g., high power + long execution time + low efficiency).

#### 🧠 Interpretation:
These patterns indicate that anomalous system behavior is typically resource-inefficient, erratic in execution time, or misaligned with expected task profiles — making the dataset well-suited for anomaly detection using both unsupervised and supervised approaches.


### 🧠 Model Approach and Reasoning
This project uses a **hybrid machine learning strategy** to detect anomalies in telemetry data. The approach combines the strengths of both **unsupervised learning** (for anomaly scoring) and **supervised learning** (for ONNX compatibility and deployment efficiency).

#### 🔍 1. Algorithm Choice
| Model Type           | Algorithm              | Purpose                                                                 |
|----------------------|------------------------|-------------------------------------------------------------------------|
| `Unsupervised Model` | `IsolationForest`      | Detect anomalies directly from data without labeled failures           |
| `Supervised Proxy`   | `RandomForestClassifier` | Trained using pseudo-labels from `IsolationForest` to mimic behavior   |

- Why Isolation Forest?
  - Efficient and effective at detecting outliers in high-dimensional telemetry data.
  - Requires no labeled training data, making it ideal for real-world edge cases.

- Why Random Forest Proxy?
  - Converts unsupervised insights into a supervised format.
  - Enables export to ONNX, allowing for optimized, low-latency edge inference.

#### 🧬 2. Features Used
- The following telemetry features were used as input to both models:
  - `cpu_usage`, `memory_usage`, `network_traffic`, `power_consumption`
  - `num_executed_instructions`, `execution_time`, `energy_efficiency`
  - `task_priority`, `task_type_compute`, `task_type_io`, `task_type_network`

#### 📊 3. Performance Metrics
| Metric            | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| `Anomaly Score`   | Output from `IsolationForest` to rank severity                    |
| `Inference Time`  | Avg time per sample measured on CPU for both models              |
| `Model Size`      | File size comparison (`.pkl` vs `.onnx`)                          |
| `ONNX Optimization` | Inference speedup (~25× faster than raw scikit-learn model)     |

Performance results showed:
- Inference time for the ONNX model: 0.000054 s/sample
- Inference time for the Isolation Forest model: 0.001299 s/sample
- ONNX model is ~3.7× larger in size, but significantly faster for real-time deployment

### ⚙️ Model Optimization and Impact
To enable efficient **edge deployment**, the scikit-learn model (`RandomForestClassifier`) was converted to **ONNX format** using skl2onnx. While `IsolationForest` is not directly supported in ONNX, we used it to generate **pseudo-labels** for training a compatible `RandomForestClassifier`. This proxy model enabled export to ONNX format and benefited from faster inference and reduced overhead in runtime environments.

#### 🚀 Optimization Summary
- **Technique**: Scikit-learn → ONNX conversion using skl2onnx
- **Target**: Reduce inference latency and improve portability to edge devices
- **Result**: Significant speed-up (25× faster) and modest increase in model size

### 📊 Performance Comparison

| Metric           | scikit-learn (.pkl) | ONNX (.onnx)       | Improvement             |
|------------------|---------------------|---------------------|--------------------------|
| File Size (KB)   | 1121.48 KB           | 470.51 KB           | ~3.7× larger             |
| Inference Time   | 0.002816 sec/sample   | 0.000026 sec/sample  | ~24× faster              |
| GPU Used         | ❌                  | ❌                  | CPU-only for both        |

#### 📌 Observations
- TThe ONNX model is **~2.4× smaller** in size and delivers a **~108× faster inference time** compared to the native `IsolationForest` implementation..
- This highlights ONNX Runtime's efficiency, especially for deployment scenarios requiring high-throughput or real-time inference.
- The ONNX model was derived from a supervised proxy (`RandomForestClassifier`) trained on labels generated by `IsolationForest`, enabling compatibility with ONNX while preserving anomaly detection logic.


### 🖥️ Integration into a Dell PC Environment

To integrate this anomaly detection model into a Dell PC, we envision the following setup:

#### 🔄 Background Service Design
- A **lightweight Python-based background service** runs periodically or continuously on the user’s PC.
- This service **collects telemetry** (e.g., CPU usage, power consumption, memory, etc.) through system APIs or lightweight agents.
- The telemetry data is then **preprocessed and passed to the ONNX model** for real-time anomaly inference.

#### 🔧 Assumptions and Environment
- **Python 3.10+** is pre-installed on the system.
- Required packages like `onnxruntime`, `psutil`, and `numpy` are bundled with the service or installed via an installer.
- The **ONNX model is precompiled and lightweight**, making it ideal for resource-constrained environments.
- A **virtual environment or embedded interpreter** can be used to avoid conflicting with system Python installations.

#### 🔁 Inference Workflow
1. Collect telemetry data at regular intervals (e.g., every 10 seconds).
2. Convert data into a float32 array and feed it into the ONNX model.
3. If the model detects an anomaly (`Warning` or `Critical`), log the event and optionally escalate.

#### 🛠️ Production Considerations
- In a production setting, this system may be **converted to C++**  for performance and native integration.
- For higher security environments, **telemetry sharing can be opt-in**, with options for `local-only` or `cloud-escalation` mode.

#### 📡 Optional Cloud Extension
- In “`share_with_dell`” mode, anomalous events may be **forwarded to Dell’s cloud** for remote diagnostics or support automation, enabled via HTTPS API calls.

This setup allows for **scalable, privacy-aware, and hardware-friendly anomaly detection** directly at the edge, helping identify issues before they affect user experience or system stability.

## 👤 Author
Seah Chen Khoon

Last updated: 25 June 2025