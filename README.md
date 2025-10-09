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

在完成 Conda 环境创建后，请从我们提供的 Google 网盘链接下载预训练模型文件。

#### 📥 Step 1. Download Pretrained Models

After setting up the environment, please download the pretrained models from the following shared Google Drive folder:  
👉 [Download Pretrained Models](https://drive.google.com/drive/folders/1JqzpNJhw3-G-Hz-OHc8qw-fjm_U4BgFd?usp=sharing)


---

#### 📂 Step 2. Place Models in the Correct Directories

After downloading, create a `model` folder under each corresponding module and place the model files accordingly:



