# OntoMap

OntoMap is a hybrid graph-learning project for exploring drug–disease relationships in the rare-disease domain. It combines a graph neural network (GNN) with drug-category similarity to rank potential drug–disease pairs for further investigation.

The repository provides a small end-to-end prototype: disease and drug data are converted into graph nodes and edges, a graph convolutional network learns node representations, and its predictions are combined with Jaccard category similarity to produce repurposing recommendations.

## Motivation

Rare diseases have limited available treatment information and often present substantial challenges in connecting patient needs with existing pharmaceutical knowledge. OntoMap explores whether combining:

- known drug–disease associations,
- drug metadata and therapeutic categories, and
- graph-structure learning

can help identify promising candidate relationships that are not yet represented in the observed dataset.

The project is intended as a research and experimental framework for rare-disease drug repurposing. The generated predictions are candidates for review; they are not clinical recommendations or validated treatment decisions.

## Repository Data

| File | Description |
|---|---|
| `Project_code` | Main Python implementation for graph construction, model training, candidate scoring, output generation, and visualization |
| `Sample_dataset_single_drug_per_row.csv` | Edge-list dataset containing one disease–drug association per row |
| `Indian Rare Diseases.xlsx` | Disease reference data, including Orphanet identifiers and disease names |
| `drugsInfo.csv.xlsx` | Drug reference data, including drug identifiers, names, and category information |

### Disease–Drug Edge Data

`Sample_dataset_single_drug_per_row.csv` has two columns:

- `ORPHAcode`: identifier or name of the rare disease
- `Available Drug IDs`: DrugBank-style identifier for an associated drug

Each row represents one observed relationship. For example:

```csv
ORPHAcode,Available Drug IDs
ORPHA:458775 (Orpha),DB00571
ORPHA:458775 (Orpha),DB00489
ORPHA:3148 (Orpha),DB00997
```

### Disease Reference Data

The disease workbook is loaded as a CSV-compatible source and provides fields including:

- `ORPHAcode`
- `Original name (your list)`

These fields are used to map internal identifiers to readable disease names in the final prediction table.

### Drug Reference Data

The drug workbook is loaded as a CSV-compatible source and provides fields including:

- `DrugID`
- `DrugName`
- `DrugCategories`

Drug categories are separated with semicolons or, where applicable, commas. They are used to compare a candidate drug with drugs already associated with a disease.

## Implementation Methodology

### 1. Reproducible Setup

The implementation fixes Python and library randomness using a seed of `42`. It initializes random number generators for Python, NumPy, and PyTorch so that experiments can be reproduced more consistently.

### 2. Data Loading and Normalization

The code loads:

- the disease–drug edge list,
- disease reference information, and
- drug metadata.

It renames the edge columns to `DiseaseID` and `DrugID`, then converts identifiers to strings to ensure consistent graph indexing and dictionary lookups.

### 3. Bipartite Graph Construction

Each unique drug and disease is represented as a graph node:

- drug nodes
- disease nodes

The observed associations become edges in a bipartite graph. A mapping from node identifiers to integer indices is used to construct a PyTorch Geometric `edge_index` tensor.

### 4. Node Feature Preparation

Drug nodes receive category-based features:

1. `DrugCategories` is split into category labels.
2. A binary category matrix is created with one column per category.
3. Known drug categories are represented by the corresponding feature row.
4. Drugs missing from the reference metadata receive zero-valued features.

Disease nodes receive zero-valued feature vectors of the same dimensionality. The resulting drug and disease feature matrices are concatenated into one node-feature tensor.

### 5. Negative Sampling

Because the dataset contains only positive disease–drug associations, equal numbers of likely-negative pairs are sampled:

- one drug node,
- one disease node,
- and no matching positive edge.

Positive and negative pairs are combined into a training edge set. Positive examples receive label `1`, while sampled negative examples receive label `0`.

### 6. Graph Neural Network

OntoMap uses a two-layer Graph Convolutional Network:

- `GCNConv(in_dim, 16)`
- `GCNConv(16, 8)`
- a linear classifier for the two possible edge labels

The node encoder applies ReLU activation after each graph convolution. For each candidate pair, it concatenates the source drug embedding and destination disease embedding and passes the concatenation through the linear classification layer.

The model is trained with:

- Adam optimizer
- learning rate: `0.01`
- cross-entropy loss
- 60 epochs

### 7. Unknown-Pair Candidate Generation

After training, the implementation generates every drug–disease combination that is not present in the known edge list. These pairs are batched for GNN inference to limit memory usage.

For each unknown pair, the model predicts the probability that the relationship is positive.

### 8. Category Similarity

For every candidate pair, the system compares the candidate drug's categories with the categories of drugs already known for the disease.

Similarity is calculated using Jaccard similarity:

```text
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

The implementation uses the maximum similarity between the candidate drug and the disease's known drugs. Missing category information is handled with a small smoothing value, while two category-empty sets receive a score of `0`.

### 9. Late Fusion

The GNN probability and category similarity score are combined using weighted late fusion:

```text
final_score = 0.8 × gnn_probability + 0.2 × category_similarity
```

This gives more weight to learned graph relationships while preserving a category-based signal from the drug metadata.

### 10. Ranking and Output

Candidate pairs are sorted by descending `final_score`. The implementation creates `category_fusion_new_predictions.csv` with:

- `DrugID`
- `DiseaseID`
- `gnn_prob`
- `cat_sim`
- `final_score`
- `DrugName`
- `DiseaseName`
- `KnownDrugsForDisease`

It also plots the 20 highest-ranked predictions as a horizontal bar chart.

## Requirements

The implementation requires:

- Python 3
- pandas
- NumPy
- PyTorch
- PyTorch Geometric
- Matplotlib

An example installation is:

```bash
pip install pandas numpy torch torch-geometric matplotlib openpyxl
```

## Usage

The current implementation uses Google Colab-style absolute paths:

```python
/content/Sample_dataset_single_drug_per_row.csv
/content/Indian Rare Diseases.xlsx - Sheet1.csv
/content/drugsInfo.csv
```

Before running it locally or in another notebook, update those paths to the locations of the repository files. In a local environment, the paths may resemble:

```python
edges = pd.read_csv("Sample_dataset_single_drug_per_row.csv")
diseases = pd.read_csv("Indian Rare Diseases.xlsx - Sheet1.csv")
drugs = pd.read_csv("drugsInfo.csv")
```

Run the `Project_code` implementation in a Python or notebook environment. It will:

1. train the graph model,
2. generate candidate predictions,
3. save `category_fusion_new_predictions.csv`, and
4. display a plot of the top 20 predictions.

## Limitations

- Negative examples are randomly sampled and may not represent true confirmed non-associations.
- Disease features are initially all-zero vectors, so graph structure and drug features carry most of the model information.
- Identifier and metadata coverage may be incomplete.
- Category similarity is a heuristic and does not establish therapeutic validity.
- Predictions should be interpreted as experimental ranking signals, not clinical evidence.

## Contributing

Contributions are welcome. Suggested improvements include adding data validation, configurable file paths, train-validation-test splits, more principled negative sampling, proper evaluation metrics, and documented experiment settings.
