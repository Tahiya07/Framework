# Unified Results Table

| evidence_area | protocol | setting | model | primary_metric | primary_value | accuracy | within_one_level | severe_error | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cognitive robustness | Bloom baseline comparison | 15% stratified hold-out (figshare_combined_dataset.csv, random_state=42) (n=?) | TF-IDF + LinearSVC | macro_f1 | 0.826 | 0.839 | 0.916 | 0.084 | bloom_evaluation.py on shared 15% hold-out split |
| cognitive robustness | Bloom baseline comparison | 15% stratified hold-out (figshare_combined_dataset.csv, random_state=42) (n=?) | Qwen2.5 zero-shot (GGUF) | macro_f1 | 0.369 | 0.441 | 0.686 | 0.314 | bloom_evaluation.py on shared 15% hold-out split |
| cognitive robustness | Bloom baseline comparison | Official Figshare test split (evaluation_results from predict_bloom / training eval) (n=2330) | Qwen2.5 LoRA (trained) | macro_f1 | 0.721 | 0.748 | 0.880 | 0.064 | bloom_evaluation.py on shared 15% hold-out split |
| cognitive robustness | Bloom classification (full test) | Figshare official test split | Qwen2.5 LoRA (full test eval) | macro_f1 | 0.721 | 0.748 | 0.880 | 0.064 | evaluation_results/metrics.json from train_qwen_bloom.py |
| student learning (RAG) | academic QA smoke benchmark | FAISS + Qwen GGUF | academic_qa SLM | token_f1_mean | 0.914 | 0.800 |  |  | small grounded QA set (evaluate_qwen_rag.py) |
| student learning (RAG) | retrieval hit@3 | FAISS + MiniLM | PrivacyRetriever | hit_at_3_mean | 1.000 |  |  |  | retrieval supports answer context |
| multimodal ingestion | PDF + image RAG smoke | PyMuPDF + OCR path | MultiModalAcademicRAG | answer_accuracy_mean | 1.000 |  |  |  | extractive multimodal smoke test |
| multimodal ingestion | multimodal_rag_smoke_v1 | pdf + image | MultiModalAcademicRAG | answer_accuracy_on_ok_cases | 0.000 |  |  |  | PDF RAG uses native text extraction from a generated text PDF.; Image RAG requires a working OCR backend; backend-unavailable is reported explicitly. |
| multimodal ingestion | OCR pipeline readiness | Tesseract / fallback | unknown | available | 0.000 |  |  |  |  |
| privacy-constrained deployment | student attack block rate | adversarial prompt taxonomy | PrivacyGuard | block_rate | 1.000 |  |  |  | measured under defined attack prompts |
| privacy-constrained deployment | student benign allow rate | benign study prompts | PrivacyGuard | allow_rate | 0.867 |  |  |  | utility under non-adversarial student queries |
| privacy-constrained deployment | attack taxonomy | direct_reconstruction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | indirect_leakage | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | model_aware_jailbreak | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | paraphrase_probe | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | partial_span_extraction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | semantic_reconstruction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | privacy baseline ablation | student attacks | no_guard | attack_block_rate | 0.000 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | role_only_no_output_guard | attack_block_rate | 0.000 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | regex_only | attack_block_rate | 0.524 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | federated_dp_only | attack_block_rate | 1.000 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | learned_plus_overlap | attack_block_rate | 1.000 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | full_hybrid_guard | attack_block_rate | 1.000 |  |  |  | compare guard variants on shared attack suite |
| federated privacy layer | federated privacy-risk model | aggregate-only updates | FederatedPrivacyGuard | attack_block_rate | 1.000 |  |  |  | no raw teacher items sent to server; parameters only |
| federated privacy layer | federated privacy-risk model | benign student prompts | FederatedPrivacyGuard | benign_allow_rate | 0.000 |  |  |  | utility after federated guard training |
