# Task-2-News_Classification-lab

## 📋 Project Overview:

### *Task 2: Transformer News Classification* 🤖
- Fine-tunes transformer models (RoBERTa, DeBERTa, or ModernBERT)
- Trains on AG News dataset (4 categories: World, Sports, Business, Sci/Tech)
- Evaluates with proper train/validation/test split (70/15/15)
- Generates performance visualizations and metrics

### *Bonus: Cross-Domain Classification* 🎁
- Applies trained model to RPP news articles
- Compares model predictions with LLM classifications
- Analyzes domain adaptation challenges

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge above
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells sequentially

### Option 2: Local Setup
bash
# Clone the repository
git clone https://github.com/yourusername/news-intelligence-lab.git
cd news-intelligence-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook


---

## 📦 Installation

### Requirements
- Python 3.10+
- CUDA-compatible GPU (recommended for Task 2)
- 8GB+ RAM

### Dependencies
bash
pip install feedparser tiktoken sentence-transformers chromadb \
    langchain langchain-community datasets transformers torch \
    matplotlib pandas seaborn scikit-learn accelerate


See [requirements.txt](requirements.txt) for exact versions.

---

## 📂 Repository Structure


news-intelligence-lab/
│
├── notebooks/
│   └── news_intelligence_lab.ipynb    # Main Jupyter notebook
│
├── data/
│   └── rpp_classified.json            # RPP news with classifications (generated)
│
├── outputs/
│   ├── model_performance.png          # F1 score comparison chart
│   └── rpp_classification.png         # RPP category distribution
│
├── models/
│   └── best_model/                    # Saved fine-tuned model
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── LICENSE                            # MIT License


---

## 🔬 Methodology

### Task 1: News Retrieval Pipeline

*Step 0: Data Loading*
- Fetch 50 latest articles from RPP RSS feed
- Extract: title, description, link, published date

*Step 1: Tokenization*
- Use tiktoken (GPT-4 encoding) for token analysis
- Calculate token counts to determine chunking needs

*Step 2: Embedding Generation*
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384
- Generate embeddings for full text (title + description)

*Step 3: Vector Storage*
- Store embeddings in ChromaDB collection
- Enable similarity-based retrieval

*Step 4: Query & Retrieval*
- Perform semantic search with Spanish queries
- Return top-k most relevant articles

*Step 5: LangChain Orchestration*
- Build modular pipeline: Load → Tokenize → Embed → Store → Retrieve
- Enable end-to-end automation

---

### Task 2: Transformer Classification

*Data Preparation*
- Dataset: AG News (120k train, 7.6k test)
- Split: 70% train / 15% validation / 15% test
- *Test set locked until final evaluation*

*Model Training*
- Architecture: RoBERTa-base (125M parameters)
- Fine-tuning on 4-class classification
- Optimization: AdamW with learning rate 2e-5
- Epochs: 3
- Batch size: 16

*Evaluation Metrics*
- F1 Score (Macro)
- F1 Score (Weighted)
- Confusion Matrix
- Per-class Precision/Recall

*Training Time*
- With GPU (T4): ~15-20 minutes (10k subset)
- Without GPU: ~6-8 hours (full dataset)

---

## 📊 Results

### Model Performance (AG News Test Set)

| Model | F1 Macro | F1 Weighted | Test Loss |
|-------|----------|-------------|-----------|
| RoBERTa-base | 0.9XXX | 0.9XXX | 0.XXX |

Note: Results will vary based on random seed and training configuration

### Classification Examples

*Sample RPP Article Classification:*

Title: "Innovación en inteligencia artificial transforma industrias"
Predicted: Science/Technology ✅
Confidence: 0.94


---

## 🎯 Key Features

✅ *Real-time News Ingestion* - Fetches latest articles from RSS feeds  
✅ *Semantic Search* - Vector-based similarity matching with ChromaDB  
✅ *State-of-the-art Models* - Fine-tuned transformers (RoBERTa/DeBERTa/ModernBERT)  
✅ *Proper ML Pipeline* - No test set snooping, proper validation  
✅ *Visualization* - Performance charts and confusion matrices  
✅ *Production-ready* - Modular code, error handling, logging  
✅ *Reproducible* - Fixed random seeds, documented dependencies  

---

## 🛠 Configuration

### Speed Optimization (Cell 9)
python
# Fast training (15-20 min) - Recommended for testing
USE_SUBSET = True  # Uses 10k samples

# Full training (6-8 hours) - For final results
USE_SUBSET = False  # Uses full 99k samples


### Model Selection (Cell 10)
python
# Choose your transformer model
model_name = "roberta-base"              # Default
# model_name = "microsoft/deberta-v3-base"  # Alternative
# model_name = "answerdotai/ModernBERT-base"  # Alternative


### Training Parameters (Cell 11)
python
num_train_epochs=3           # Number of training epochs
learning_rate=2e-5           # Learning rate
per_device_train_batch_size=16  # Batch size (reduce if OOM)


---

## 📈 Performance Tips

### To Speed Up Training:
1. ✅ *Enable GPU* in Colab (Runtime → Change runtime type → T4 GPU)
2. ✅ *Use subset* (USE_SUBSET = True in Cell 9)
3. ✅ *Reduce epochs* (Change num_train_epochs=1 in Cell 11)
4. ✅ *Increase batch size* if you have GPU memory (e.g., 32)

### To Improve Accuracy:
1. 📈 Train on *full dataset* (USE_SUBSET = False)
2. 📈 Increase *epochs* to 5
3. 📈 Try *different models* (DeBERTa often outperforms RoBERTa)
4. 📈 Tune *hyperparameters* (learning rate, weight decay)

---

## 🧪 Experiments & Extensions

### Bonus Task: LLM Comparison
python
# Use OpenAI API to classify RPP articles
import openai

def classify_with_llm(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Classify this news into World/Sports/Business/Sci-Tech: {text}"
        }]
    )
    return response.choices[0].message.content

# Compare with model predictions


### Multi-Model Comparison
Train all three models and compare:
- RoBERTa-base
- DeBERTa-v3-base  
- ModernBERT-base

---

## 📝 Evaluation Rubric (20 pts)

| Category | Points | Criteria |
|----------|--------|----------|
| *Data & Reproducibility* | 4 | ✅ Organized structure, functional notebook, requirements.txt, relative paths |
| *Task 1: Retrieval* | 6 | ✅ RSS parsing, tokenization, embeddings, ChromaDB, LangChain, query results |
| *Task 2: Classification* | 6 | ✅ AG News split (70/15/15), model training, no test snooping, F1 scores |
| *Visualization* | 2 | ✅ F1 comparison chart, proper labels, interpretation |
| *Bonus* | +3 | ✅ LLM classification, comparison analysis, F1 scores vs LLM |

---

## 🐛 Troubleshooting

### Common Issues

*1. GPU Not Available*

⚠  CUDA available: False

*Solution:* Enable GPU in Colab (Runtime → Change runtime type → T4 GPU)

---

*2. Training Too Slow*

[522/18540 12:47 < 7:23:16]

*Solution:* Set USE_SUBSET = True in Cell 9 to reduce dataset size

---

*3. Out of Memory Error*

RuntimeError: CUDA out of memory

*Solution:* Reduce batch size in Cell 11: per_device_train_batch_size=8

---

*4. Model Not Found Error*

NameError: name 'model' is not defined

*Solution:* Run cells in order. Re-run Cell 10 to reload the model.

---

*5. ChromaDB Collection Exists*

Collection already exists

*Solution:* The code handles this automatically (deletes and recreates)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request


## 🙏 Acknowledgments

- *Hugging Face* for transformers library and pre-trained models
- *RPP Perú* for providing RSS feed
- *AG News* dataset for classification task
- *LangChain* for orchestration framework
- *ChromaDB* for vector database
