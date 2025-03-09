# Dynagh: Dynamic Geometric Data Creation for Geometric Neural Networks

Dynagh is a comprehensive research and development tool designed for the dynamic creation of geometric (graph) data from tabular (CSV) inputs. The application transforms raw CSV data into structured graph representations with rich feature embeddings—using techniques such as BERT, GloVe, or Word2Vec—that can be fed into geometric neural networks (GNNs). The tool also provides an interactive frontend for configuring nodes and relationships and visualizing the resulting graphs.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Codebase Structure](#codebase-structure)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#usage)
  - [Running the Backend](#running-the-backend)
  - [Running the Frontend](#running-the-frontend)
  - [Workflow Overview](#workflow-overview)
- [Advanced Configuration](#advanced-configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

Dynagh is built to assist researchers and developers in creating and experimenting with geometric data for neural network applications. The tool:
  
- **Transforms CSV Data:** Converts tabular data into graphs using configurable node and relationship mappings.
- **Generates Feature Embeddings:** Uses a variety of embedding methods (BERT, GloVe, Word2Vec) and processing techniques (standardization, normalization, dimensionality reduction via PCA/UMAP) to attach rich feature representations to nodes.
- **Supports GNN Training:** Provides a suite of GNN architectures—including GCN, GraphSAGE, GAT, GIN, ChebConv, and a Residual GCN—as well as a Naive Bayes baseline for node classification, complete with training, evaluation, visualization, and ensemble methods.
- **Offers an Interactive Frontend:** A React-based interface lets users upload CSV files, select configuration options for graph creation, and visualize the resulting network via interactive graph layouts and React Flow.

This tool is ideal for anyone looking to dynamically generate and process graph data for research in geometric neural networks or related fields.

---

## Features

- **Dynamic Graph Construction:** Automatically builds graph structures from CSV data based on user-defined node and relationship configurations.
- **Feature Space Creation:** Supports advanced feature engineering with text preprocessing, embedding generation, and dimensionality reduction.
- **Multiple Embedding Methods:** Choose between BERT, GloVe, or Word2Vec for generating text embeddings.
- **GNN Architectures:** Implements multiple GNN models along with a Naive Bayes baseline for comparative studies.
- **Visualization & Interaction:** Interactive React frontend using React Flow and force-directed graphs for real-time configuration and analysis.
- **Robust Logging & Error Handling:** Detailed logging throughout the backend helps track data processing, feature generation, and graph construction.
- **Customizable & Extensible:** Modular codebase that can be easily extended to include additional processing steps or new model architectures.

---

## Codebase Structure

The repository is divided into two main parts:

### Backend

- **`DataFrameToGraph.py`**  
  Converts pandas DataFrames into NetworkX graph objects based on a user-provided configuration for nodes and relationships.

- **`FeatureSpaceCreator.py`**  
  Processes text and numeric data, creates embeddings using various methods, applies preprocessing and dimensionality reduction, and attaches these features to graph nodes.

- **`main.py`**  
  Implements a FastAPI server with a `/process-data` endpoint that accepts CSV data and configuration, builds the graph, attaches features, and returns the final graph data (as JSON) and CSV representation of feature data.

- **`TorchGeometricGraphBuilder.py`**  
  Parses JSON graph data into a PyTorch Geometric Data object. This module also includes multiple GNN model implementations (GCN, GraphSAGE, GAT, GIN, ChebConv, ResidualGCN), training routines with early stopping, evaluation metrics, and visualization methods.

### Frontend

- **React Application:**  
  Provides a rich, interactive user interface built with React. Key components include:
  - **FileUploader:** For CSV file upload and parsing.
  - **ConfigurationPanel:** To select node columns, define relationships, and configure advanced feature creation.
  - **GraphNet & ReactFlowWrapper:** For visualizing and manipulating the graph layout.
  - **GraphVisualizer & Sidebar:** To display graph statistics, node degree distribution, and offer download options.
  - **Modals (NodeEditModal, RelationshipModal, InfoModal):** For editing node properties, defining relationships, and displaying additional information.
  
- **Supporting Files:**  
  - `package.json`, CSS styles, and various hooks (e.g., `useGraph.js`) that manage state and API interactions with the backend.

---

## Installation

### Prerequisites

- **Python 3.8+** (with pip)
- **Node.js** (v14+ recommended)
- **Git Bash for Windows** (recommended for command-line operations)

### Backend Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/Dynagh.git
   cd Dynagh/backend
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Use 'source venv/bin/activate' on Unix systems
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables (if needed):**  
   For example, set paths for caching GloVe models or specifying device preferences.

### Frontend Setup

1. **Navigate to the Frontend Directory:**

   ```bash
   cd ../frontend
   ```

2. **Install Node Dependencies:**

   ```bash
   npm install
   ```

3. **Configure API URL (optional):**  
   If you need to change the backend URL, set the `REACT_APP_API_BASE_URL` variable in a `.env` file.

---

## Usage

### Running the Backend

From the backend directory in Git Bash, run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

This starts the FastAPI server, exposing the `/process-data` endpoint for processing CSV data and building graphs.

### Running the Frontend

From the frontend directory in Git Bash, run:

```bash
npm start
```

This will launch the React application in your default web browser, allowing you to interact with the tool.

### Workflow Overview

1. **CSV Upload:**  
   Use the File Uploader in the frontend to drag and drop your CSV file. The file parser (via PapaParse) extracts data and headers.

2. **Configuration:**  
   In the Configuration Panel, select which CSV columns should be treated as nodes and optionally define relationships between them. Enable advanced feature creation if you wish to generate embeddings (using BERT, GloVe, or Word2Vec).

3. **Graph Processing:**  
   Upon clicking "Process Graph," the frontend sends your data and configuration to the backend. The backend:
   - Converts the CSV data into a graph.
   - Attaches precomputed or dynamically generated feature embeddings.
   - Optionally processes the graph with Torch Geometric routines.

4. **Visualization & Analysis:**  
   The resulting graph data is returned and visualized through interactive components. You can inspect nodes, adjust relationships, and download the processed graph in JSON or CSV format.

5. **GNN Training & Evaluation:**  
   The backend includes modules to train various GNN architectures on the generated graph data, complete with validation, testing, and visualization of node embeddings.

---

## Advanced Configuration

- **Feature Embedding Configuration:**  
  Define custom feature configurations in the frontend to specify parameters such as:
  - **Embedding Method:** BERT, GloVe, or Word2Vec.
  - **Embedding Dimensions:** e.g., 768 for BERT.
  - **Preprocessing Options:** Stopwords removal, text cleaning, and tokenization settings.
  - **Dimensionality Reduction:** Apply PCA or UMAP if needed.

- **GNN Model Selection:**  
  The backend supports multiple architectures. You can adjust training parameters (learning rate, dropout, epochs, etc.) via command-line arguments when launching the training script (see `TorchGeometricGraphBuilder.py`).

- **Logging & Debugging:**  
  Detailed logging is implemented across the backend. Check log files (e.g., `logs/feature_space_creator.log`) for insights into the processing pipeline.

---

## Contributing

Contributions are welcome! If you wish to extend the functionality of Dynagh or fix bugs:
- Fork the repository.
- Create a feature branch.
- Commit and push your changes.
- Submit a pull request with a detailed description of your changes.

Please ensure that your contributions follow the existing code style and include appropriate tests and documentation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or further discussion, please contact:

- **Author:** Your Name  
- **Email:** your.email@example.com  
- **GitHub:** [your-username](https://github.com/your-username)

---

Dynagh provides a full-stack solution for the dynamic creation and analysis of geometric data tailored for GNN applications. With its robust backend processing and intuitive frontend interface, this tool serves as a valuable resource for research and experimentation in geometric neural networks. Enjoy exploring and expanding its capabilities!

