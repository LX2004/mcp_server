# Paper"LLMs as active designers for synthetic biology via the Model Context Protocol"

---
<img width="865" height="593" alt="image" src="https://github.com/user-attachments/assets/3ba15fbe-5ded-4ca0-a101-25cc166081e7" />
---

## Overview


This repository demonstrates how large language models (LLMs) can be integrated with the **Model Context Protocol (MCP)** to automate tasks in the **Design–Build–Test–Learn (DBTL)** cycle of synthetic biology.  
The system enables the LLM to interpret natural-language prompts, delegate computational tasks to MCP-connected modules, and orchestrate workflows for applications such as **CRISPRi gRNA design**, **promoter sequence optimization**, and **RNA switch design**.  
Through this integration, LLMs evolve from passive text generators into **active AI designers**, capable of coordinating domain-specific tools to perform complex biological design and analysis.


---

### 🧬 `fitness`
**CRISPRi gRNA Optimization**

This module focuses on the **design and optimization of guide RNAs (gRNAs)** for CRISPR interference (CRISPRi).  
It performs comprehensive analyses including:
- On-target efficiency prediction  
- Off-target assessment  
- Binding energy computation  
- Fitness estimation in microbial systems  

By integrating these steps, the module identifies high-specificity, high-efficiency gRNAs and can be invoked directly through MCP-driven LLM commands.

---

### 🧫 `promoter`
**Promoter Sequence Optimization and Design**

This module implements **generative and predictive modeling** for promoter sequence engineering.  
It operates in an iterative design loop that generates candidate promoters, predicts their transcriptional strength, and refines the sequences based on optimization feedback.  
This allows for the creation of promoters tuned to desired expression levels or multi-gene regulatory architectures, all within an autonomous LLM–MCP framework.

---

### 🧠 `toehold`
**RNA Toehold Switch Design**

This module enables **rational design of RNA-based switches**, a key regulatory element in synthetic biology.  
It evaluates candidate sequences based on:
- Thermodynamic stability  
- Base-pairing energy  
- Trigger specificity and response potential  

The system can autonomously generate and rank RNA switches, guided entirely by natural-language prompts through the MCP interface.

---

### 💡 Summary
Together, these modules constitute a unified framework for intelligent biological design.  
By coupling **large language models** with **domain-specific computation via MCP**, this repository transforms LLMs from passive reasoning systems into **active AI designers**—capable of performing CRISPRi optimization, promoter engineering, and RNA switch design with minimal human intervention.  
This approach exemplifies a step toward **autonomous, self-directed synthetic biology**, aligning with the forward-looking principles of next-generation DBTL automation.

---

## 🧬 Environment Configuration and Using Guide

To ensure reproducibility of the experiments, all dependencies and software versions are stored in the `environment.yml` file.

---

### 🔧 1. Create Environment

```bash
conda env create -f environment.yml

conda activate mcp_server
```
### 🧠 2.Pretrained Models Setup

After creating the Conda environment, please download the pretrained models from our shared Google Drive link.


#### 📥 Step 1. Download Pretrained Models

After setting up the environment, please download the pretrained models from the following shared Google Drive folder:  
👉 [Download Pretrained Models](https://drive.google.com/drive/folders/1JqzpNJhw3-G-Hz-OHc8qw-fjm_U4BgFd?usp=sharing)


---

#### 📂 Step 2. Place Models in the Correct Directories

Once downloaded, place the models in their respective module directories:

- **Promoter model:** Place the pretrained promoter model inside a new folder named `model` under the `promoter` directory.  
  Example path: `promoter/model/promoter_model.pth`

- **RNA switch model:** Place the pretrained RNA switch model inside a new folder named `model` under the `rna_switch` directory.  
  Example path: `rna_switch/model/rna_switch_model.pth`

### 🚀 3. Run and Verify Each Synthetic Biology Task

After completing the environment setup and placing pretrained models, run the following scripts to verify functionality for each module.  
Each task can be executed independently using the same workflow:

#### Task 1: CRISPRi gRNA Optimization

```bash
cd fitness/code
python main_optimization_grna.py

```
---

#### Task 2: Promoter Optimization Design
```bash
cd promoter/code
python opt_promoter_main.py

```

---

#### Task 3: RNA Switch Design
```bash
cd rna_switch/code
python main_multi_object_opt.py


```

---


## MCP Server Installation
For detailed steps, please refer to the [link](https://modelcontextprotocol.io/quickstart/server#why-claude-for-desktop-and-not-claude-ai).

### 🖥️ System Requirements

Before running the project, please ensure the following system requirements are met:

- **Python 3.10 or higher** must be installed.  
- **Python MCP SDK 1.2.0 or higher** is required.

It is recommended to use **Conda** for environment management to maintain version consistency across dependencies.


### UV Installation
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, ensure `uv` is available in your terminal:

```bash
uv --version
```


###  MCP Server Project (Local deployment)




1. Download the Server Poject File from:
2. 在本地解压缩
3. 打开压缩后的文件夹，目录如下：


```text
bio-mcp/
├── rna_main_mcp.py              # MCP server entry point
├── bio.py                       # Core bio-related logic
├── rna_utils.py                 # RNA utilities
├── rna_unet.py                  # RNA U-Net model
├── rna_diffusion.py             # Diffusion models
├── opt_promoter_main.py         # Promoter optimization
├── promoter_utils*.py           # Promoter utilities
├── fitness.py                   # Fitness calculations
├── calculation_off_target.py    # Off-target calculations
├── pyproject.toml               # Project & dependency configuration
├── uv.lock                      # uv lockfile
├── .python-version              # Optional Python version hint
├── net/                         # Model weights (if present)
└── README.md                    # This file
```


4. 









### Create a new directory for our project
```bash
uv init fitness_prediction
cd fitness_prediction
```

### Create a virtual environment and activate it
```bash
uv venv
.venv\Scripts\activate.bat
```

### Install dependencies
```bash
uv add "mcp[cli]" httpx
uv add numpy scikit-learn scipy xgboost pandas matplotlib
```

### Create our server file
```bash
touch fitness_prediction.py
```
Copy the contents of `MCP/prediction.py` to the created `fitness_prediction.py` file.

All files:
![image](https://github.com/user-attachments/assets/df02712e-56bb-47b1-8f5f-f1f23d1668d7)

##  Using MCP Server
### Install third-party software
[cherry](https://www.cherry-ai.com/) or [Cursor](https://www.cursor.com/cn).

### MCP Profile
Replace "ABSOLUTE_PATH_PLACEHOLDER" in the `MCP/mcp_config.json` file with the absolute path of the project.

After the replacement is completed, you can query the running MCP server in the third-party software.

## 4. Making RAG by cherry studio
```bash
python download_abstract.py
```


