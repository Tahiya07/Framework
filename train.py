import json, pandas as pd, numpy as np, time, re, ast
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from langdetect import detect, DetectorFactory

# Set seed for consistent language detection
DetectorFactory.seed = 42

# 1. SEMANTIC CLEANING & FILTERING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_english(text):
    try:
        # filter for English
        return len(text) > 8 and detect(text) == 'en'
    except:
        return False

# 2. DATA HARMONIZATION
def load_and_standardize():
    print("Loading and cleaning datasets...")
    
    # --- Mendeley Files (Correcting to plural 'Questions') ---
    f1 = pd.read_csv('data/Data_Structure.csv')
    f2 = pd.read_csv('data/Introduction_to_Computers_and_Research.csv')
    m_map = {5:'Remembering', 10:'Understanding', 15:'Applying', 20:'Analyzing', 30:'Creating'}
    df_acad = pd.concat([f1, f2])
    df_acad['label'] = df_acad['Score'].map(m_map)
    df_acad = df_acad[['Questions', 'label']].rename(columns={'Questions': 'text'})

    # --- Irrelevant Questions (Noise) ---
    df_noise = pd.read_csv('data/Irrelevant_Questions.csv')
    df_noise['label'] = 'Irrelevant'
    df_noise = df_noise[['Questions', 'label']].rename(columns={'Questions': 'text'})

    # --- MOOC-Radar (problem.json as JSONL) ---
    mooc_list = []
    mooc_map = {1:'Remembering', 2:'Understanding', 3:'Applying', 4:'Analyzing', 5:'Evaluating', 6:'Creating'}
    with open('data/problem.json', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                # Unpack noisy stringified dict in 'detail' field
                detail = ast.literal_eval(item['detail'])
                content = detail.get('content', '')
                cog_dim = item.get('cognitive_dimension')
                
                # Filter for English content only
                if content and is_english(content) and cog_dim in mooc_map:
                    mooc_list.append({'text': content, 'label': mooc_map[cog_dim]})
            except: continue
    df_mooc = pd.DataFrame(mooc_list)

    # Merge, Clean, and Final Label Standardizing
    df_full = pd.concat([df_acad, df_noise, df_mooc]).dropna()
    df_full['text'] = df_full['text'].apply(clean_text)
    df_full['label'] = df_full['label'].replace('Evaluating', 'Analyzing') # Grouping for higher F1
    
    valid_labels = ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Creating', 'Irrelevant']
    df_full = df_full[df_full['label'].isin(valid_labels)]
    
    print(f"Total processed samples: {len(df_full)}")
    return train_test_split(df_full['text'], df_full['label'], test_size=0.2, random_state=42)

# --- EXECUTION & MODELS ---
X_train, X_test, y_train, y_test = load_and_standardize()

# Semantic Baseline SVM (Under 100MB)
print("\nExtracting Semantic Embeddings (MiniLM-L6)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2') 
X_train_vec = embedder.encode(X_train.tolist())
X_test_vec = embedder.encode(X_test.tolist())

svm_model = SVC(kernel='rbf', C=10, class_weight='balanced').fit(X_train_vec, y_train)

# Qwen Target (Local Privacy-Preserving GGUF)
print("\nInitializing Qwen 2.5 1.5B (GGUF)...")
llm = Llama(model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf", n_ctx=2048, n_threads=4)

def qwen_classify(text):
    # This specific prompt structure is optimized for Qwen 2.5 1.5B
    prompt = f"""<|im_start|>system
You are a university assistant. Use ONLY one word from this list: [Remembering, Understanding, Applying, Analyzing, Evaluating, Creating, Irrelevant].
Rule: If it's a greeting or not about course content, say 'Irrelevant'.<|im_end|>
<|im_start|>user
Query: "{text}"
Label:<|im_end|>
<|im_start|>assistant
Label: """

    try:
        # Temperature 0 is critical for accuracy in 1.5B models
        response = llm(prompt, max_tokens=10, temperature=0, stop=["<|im_end|>", "\n"], echo=False)
        
        # CORRECT INDEXING: ['choices'][0]['text']
        res = response['choices'][0]['text'].strip().capitalize()
        
        # Expert Fuzzy Matcher: Catch the label even if the model adds "The label is..."
        valid_labels = ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating', 'Irrelevant']
        for L in valid_labels:
            if L in res:
                return L
                
        return "Unknown"
    except Exception as e:
        # Log the error for debugging (helpful for paper documentation)
        return f"Error: {str(e)}"


# --- RESEARCH AUDIT LOOP ---
print("\nExecuting Qwen Accuracy Audit...")
y_pred_qwen = []
subset_size = 50 # Small subset for quick latency check
test_subset = X_test.iloc[:subset_size]
y_true_subset = y_test.iloc[:subset_size]

start_time = time.time()
for i, t in enumerate(test_subset):
    pred = qwen_classify(t)
    y_pred_qwen.append(pred)
    # Print the first 5 to see if the model logic is working
    if i < 5:
        print(f"Sample {i} | Q: {t[:30]}... | Pred: {pred}")

qwen_latency = (time.time() - start_time) / subset_size
print(f"\nTARGET: QWEN 2.5 1.5B (Latency: {qwen_latency:.4f}s)")
print(classification_report(y_true_subset, y_pred_qwen, zero_division=0))



# --- RESEARCH AUDIT (LATENCY & F1) ---
print("\n" + "="*45 + "\nFINAL RESEARCH METRICS\n" + "="*45)

# SVM Report (Already printed in your logs, but here for completeness)
y_pred_svm = svm_model.predict(X_test_vec)
print("BASELINE: SEMANTIC SVM\n", classification_report(y_test, y_pred_svm, zero_division=0))

# Qwen Report (Corrected Loop)
print("Generating Qwen Predictions...")
start_time = time.time()
y_pred_qwen = []
# Use a smaller slice if testing for latency (e.g., first 50)
test_subset = X_test.iloc[:50] 
y_true_subset = y_test.iloc[:50]

for t in test_subset:
    y_pred_qwen.append(qwen_classify(t))

qwen_latency = (time.time() - start_time) / len(test_subset)

print(f"\nTARGET: QWEN 2.5 1.5B (Latency: {qwen_latency:.4f}s per query)\n")
print(classification_report(y_true_subset, y_pred_qwen, zero_division=0))

import csv

def save_research_metrics(y_true, y_pred_svm, y_pred_qwen, filename="research_audit.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Query_ID", "Actual_Label", "SVM_Prediction", "Qwen_Prediction"])
        for i in range(len(y_pred_qwen)):
            writer.writerow([i, y_true.iloc[i], y_pred_svm[i], y_pred_qwen[i]])
    print(f"Research metrics exported to {filename}")

# Call this after your loop
save_research_metrics(y_test[:50], y_pred_svm[:50], y_pred_qwen)
