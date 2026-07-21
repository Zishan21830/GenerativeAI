<div align="center">

# 🦜🔗 LangChain Mastery Hub

**A highly structured, hands-on repository tracking my journey and deep dives into building production-ready LLM applications using LangChain.**

[![Build Status](https://shields.io)](#)
[![License: MIT](https://shields.io)](#)
[![Python Version](https://shields.io)](#)
[![LangChain Version](https://shields.io)](#)

<br>
<img src="https://githubusercontent.com" alt="LangChain Project Banner" width="100%" max-width="800px" />
<br>

</div>

---

## 📌 Table of Contents
* [📖 About the Project](#-about the-project)
* [🚀 Key Features & Modules](#-key-features--modules)
* [🛠️ Tech Stack](#️-tech-stack)
* [📥 Getting Started](#-getting-started)
* [⚙️ Configuration](#️-configuration)
* [👥 Contributors](#-contributors)

---

## 📖 About the Project

This repository serves as a centralized, production-grade learning sandbox dedicated to mastering the **LangChain ecosystem**. It contains practical implementations, modular code syntax, and clean architecture designs tracking complex LLM design patterns. The project explores everything from connecting multi-provider foundation models to enforcing strict, schema-driven structured JSON outputs for reliable automation.

---

## 🚀 Key Features & Modules

### 🤖 1. Model Providers & Integrations
* **Closed-Source Ecosystem**: Production setups using **OpenAI**, **Anthropic Claude**, and **Google Gemini**.
* **Open-Source Architectures**: Local and remote inference strategies via **HuggingFace** pipelines.
* **Unified Interface**: Implementations spanning standard `LLMs`, stateful `ChatModels`, and mathematical vector `Embeddings`.

### 📝 2. Advanced Prompt Engineering
* **Template Generation**: Scalable implementations of static and dynamic `PromptTemplates`.
* **Contextual Framing**: Production-grade configurations utilizing multi-role message arrays (`SystemMessage`, `HumanMessage`, `AIMessage`).

### 🧩 3. Strict Structured Outputs
* **Schema Enforcement**: Programmatic data extraction utilizing multiple data formats.
* **Typing Systems**: Robust implementations leveraging Python **TypedDict**, **Pydantic (v2) Data Models**, and raw **JSON Schema** dictionaries natively matched inside `.with_structured_output()`.

---

## 🛠️ Tech Stack

* **Core Framework**: LangChain / LangChain-Community
* **Language Runtime**: Python 3.10+
* **Data Validation**: Pydantic v2
* **Model Gateways**: OpenAI API, Anthropic API, Google GenAI, HuggingFace Hub

---

## 📥 Getting Started

<details>
<summary>📋 <b>View Prerequisites & System Setup</b></summary>

Before initializing the workspace, ensure your machine meets the following environment dependencies:
* Python `3.10` or higher installed globally.
* Active access tokens/API credentials for your chosen LLM providers.
* Git installed on your local operating system.
</details>

### 💻 Local Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd langchain-mastery-hub
   ```

2. **Create and activate a isolated virtual environment:**
   ```bash
   python -m venv genai
   
   # For macOS/Linux:
   source genai/bin/activate
   
   # For Windows (CMD):
   genai\Scripts\activate.bat
   ```

3. **Install the pinned project dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

Your sensitive API credentials and runtime settings must be safely compartmentalized inside a root-level environment file. 

1. Create a `.env` file in the root workspace directory:
   ```bash
   touch .env
   ```

2. Populate the `.env` file with your specific target provider keys:
   ```text
   # LLM Provider Keys
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_claude_api_key_here
   GOOGLE_API_KEY=your_gemini_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

   # LangSmith Observability (Optional but Recommended)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   ```

> ⚠️ **Security Warning**: The `.env` file contains production secrets and is explicitly blocked from tracking via our local `.gitignore` rule (`genai/` and `.env`). Do not commit it to your upstream remote public repository.

---

## 👥 Contributors

Contributions, issues, and feature requests are welcome! Feel free to fork this project and add new components as you advance.

<div align="left">
  <a href="https://github.com">
    <img src="https://unsplash.com" alt="Contributor Profile Avatar" style="border-radius: 50%; margin: 5px;" width="60px" height="60px"/>
  </a>
</div>

---