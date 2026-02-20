# Praxis DGMS

### An Interactive Decision Guidance Management System

Praxis DGMS is an open-source framework for building and running interactive decision guidance systems over Virtual Things (VTs) — parameterized products, services, or design specifications. It provides a layered infrastructure for multi-objective optimization and interactive Pareto-based decision support.

#### Praxis integrates:
- OptiGuide+ (interactive decision guidance engine)
- DGAL (Decision Guidance Analytics Language)
- DG-ViTh (A structured artifact repository)
- Mathematical optimization solvers (e.g., Gurobi)
- A reproducible project-based workflow

#### Praxis allows developers to:
- Model domain-specific Virtual Things
- Define requirements and objectives
- Run multi-objective optimization
- Launch an interactive guidance interface
- Deliver final optimal recommendations

For methodological and theoretical details, refer to the
[OptiGuide+ paper (ICEIS 2025)](https://www.scitepress.org/Papers/2025/134700/134700.pdf).

---

## Repository Structure

### 1. PraxisDGMS (System Engine)
This folder contains the core system libraries and infrastructure:

```
PraxisDGMS/
├── lib/
│   ├── dgal_lib/
│   ├── optiguide_lib/
│   └── vThings/
```

It includes:
- Core optimization logic
- DGAL execution engine
- OptiGuide interaction layer
- System utilities

This folder serves as the system engine and should not be modified for individual projects.


### 2. Project Folder (User Workspace)
Each domain-specific application is defined in a separate project folder.  
This folder can be located anywhere on your machine.

#### Required Structure
```
MyProject/
├── configs/
│   └── config.json
├── analyticModels/
├── metricSchemas/
├── vtSpecs/
├── reqSpecs/
```

The project folder defines:
- Analytic models
- Virtual Thing specifications
- Metric schemas
- Requirement specifications
- Configuration settings

An example project “Procurement” is available under:
PraxisDGMS/ProjectExample

---

## Installation


### 1. Install Anaconda or Miniconda
Download and install from:  
https://conda.io/projects/conda/en/latest/user-guide/install/index.html


### 2. Install Gurobi
Follow the installation instructions in this video and activate your Gurobi license:  
https://www.youtube.com/watch?v=ZcL-NmckTxQ


### 3. Activate Environment
Open a terminal and run:
```bash
conda activate base
```

### 4. Install Pyomo
```bash
conda install -c conda-forge pyomo
```

### 5. Clone PraxisDGMS from GitHub
```bash
git clone https://github.com/Tahani2015/PraxisDGMS.git
```
Or download the ZIP file from GitHub and extract the PraxisDGMS folder to your machine.


---


## Running Praxis


### Step 1 – Preprocessing

Navigate to the PraxisDGMS root directory, then run:

```bash
python lib/optiGuide_lib/mainPreprocessing.py --project-dir "/path/to/myProject"
```
This prepares the recommendation data and computes Pareto-optimal solutions.

### Step 2 – Launch the Interactive Interface

```bash
python lib/optiGuide_lib/optiGuideUI.py --project-dir "/path/to/myProject"
```
The graphical interface will open, and you can explore trade-offs, compare alternatives
and select final recommendations.

---
