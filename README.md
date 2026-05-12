# Paper"GeneGPT: Tool-Augmented Large Language Models Enable Autonomous Biological Design via the Model Context Protocol"

---
<img width="1804" height="1484" alt="image" src="https://github.com/user-attachments/assets/6c4cd7da-e358-465c-9188-8464215f94c1" />
---

## Overview


This repository demonstrates how large language models (LLMs) can be integrated with the **Model Context Protocol (MCP)** to automate tasks in the **Design–Build–Test–Learn (DBTL)** cycle of synthetic biology.  
The system enables the LLM to interpret natural-language prompts, delegate computational tasks to MCP-connected modules, and orchestrate workflows for applications such as **CRISPRi gRNA design**, **promoter sequence optimization**, and **RNA switch design**.  
Through this integration, LLMs evolve from passive text generators into **active AI designers**, capable of coordinating domain-specific tools to perform complex biological design and analysis.


---

### 🧬 `fitness`
**CRISPRi gRNA Optimization**

<img width="1814" height="740" alt="image" src="https://github.com/user-attachments/assets/145f89d9-d8c0-4be7-acb9-5049f2929a20" />


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

<img width="2563" height="1366" alt="image" src="https://github.com/user-attachments/assets/f9c560b8-56b6-44bc-b166-19cbfeeebf5a" />

This module implements **generative and predictive modeling** for promoter sequence engineering.  
It operates in an iterative design loop that generates candidate promoters, predicts their transcriptional strength, and refines the sequences based on optimization feedback.  
This allows for the creation of promoters tuned to desired expression levels or multi-gene regulatory architectures, all within an autonomous LLM–MCP framework.

---

### 🧠 `toehold`
**RNA Toehold Switch Design**

<img width="2044" height="281" alt="image" src="https://github.com/user-attachments/assets/74d1272e-9944-471a-aa33-a0f9835e6986" />
<img width="3243" height="1329" alt="image" src="https://github.com/user-attachments/assets/8acdaa43-c026-4063-b152-93d3efde723a" />


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




1. Download the Server Poject File from: [https://drive.google.com/drive/folders/1-QyJfEgeAT_6YG7RuxiROh1X15-F4LCU?usp=drive_link](https://drive.google.com/drive/folders/1-QyJfEgeAT_6YG7RuxiROh1X15-F4LCU?usp=drive_link)

2. Extract the archive locally to obtain the `bio`, `data`, and `model` folders, and make sure they are in the same directory.

3. Open the `bio` folder. Its contents are shown in the figure below.


<img width="893" height="955" alt="image" src="https://github.com/user-attachments/assets/e2611ed6-22d3-4474-bb04-d1c5666671b1" />


4. Open CMD in the `bio` directory

In File Explorer, go to the `bio` folder, then:

- Click the address bar, type `cmd` and press **Enter**.


Check uv it’s installed:
```bash
uv --version
```


5. Create a virtual environment and install dependencies

```bash
# 1) Create a virtual environment in .venv
uv venv

# 2) Install all dependencies based on uv.lock / pyproject.toml
uv sync
```

6. Activate the virtual environment (Windows)

```bash
.venv\Scripts\activate
```
<img width="251" height="50" alt="image" src="https://github.com/user-attachments/assets/8b2837b6-0f5f-4d80-accd-2aa6272422b3" />

7. Run your project scripts

```bash
uv run python bio.py
```

<img width="566" height="149" alt="image" src="https://github.com/user-attachments/assets/6e8cb7ec-864f-49f7-9b69-0f9309be72f4" />


##  Using MCP Server
1. Install third-party software
[cherry](https://www.cherry-ai.com/) or [Cursor](https://www.cursor.com/cn).

2. After installing Cherry Studio, open it, click the gear icon in the upper-right corner to enter the settings page, and then configure the relevant MCP server parameters.

<img width="3816" height="603" alt="f2dba37e-0dd5-4a26-8d82-a0c7a9ea83a6" src="https://github.com/user-attachments/assets/f0bc1c3d-c566-4025-893f-b8b09bd49f6d" />

3. Then copy and paste the contents of the `mcp_json.json` file from the current repository, and **be sure to change the MCP server address to your local address**.

<img width="944" height="1182" alt="image" src="https://github.com/user-attachments/assets/9cdec828-4965-4718-a3bd-3cbaeb60fa7c" />

4. Run the MCP server.

<img width="3392" height="437" alt="image" src="https://github.com/user-attachments/assets/8cedb5c8-37e4-4d1f-a4ed-5787b281a190" />

5. Create an agent and enable MCP server access for it.
<img width="1445" height="909" alt="image" src="https://github.com/user-attachments/assets/88f3d8e6-8d3e-4f5b-888e-070635f82259" />

6. For example
<img width="3368" height="1368" alt="image" src="https://github.com/user-attachments/assets/65fc6687-b268-4b7e-80df-8388511f8b0f" />



