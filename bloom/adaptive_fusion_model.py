import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _extract_cue_features(texts):
    """
    Extract Bloom cue-like signals (verbs / instruction words).
    Simple heuristic version (you can later improve this).
    """
    cue_keywords = {
        "analyze", "evaluate", "create", "explain", "describe",
        "compare", "discuss", "identify", "define", "list"
    }

    features = []
    for t in texts:
        t_low = t.lower()
        score = sum(1 for w in cue_keywords if w in t_low)
        features.append([score])
    return np.array(features)


class AdaptiveCueContentFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    ML novelty:
    Learns optimal fusion of:
        - cue features (instruction verbs)
        - content features (TF-IDF)

    Final representation:
        h = alpha * cue + (1 - alpha) * content
    """

    def __init__(self, alpha_grid=None, C=1.0):
        self.alpha_grid = alpha_grid if alpha_grid is not None else np.linspace(0, 1, 11)
        self.C = C

    def fit(self, X, y):
        self.vectorizer = TfidfVectorizer(max_features=5000)

        X_text = np.array(X)

        cue = _extract_cue_features(X_text)
        content = self.vectorizer.fit_transform(X_text).toarray()

        best_alpha = 0
        best_score = -1

        for alpha in self.alpha_grid:
            fused = np.hstack([
                alpha * cue,
                (1 - alpha) * content
            ])

            clf = LogisticRegression(max_iter=2000, C=self.C)
            clf.fit(fused, y)

            score = clf.score(fused, y)

            if score > best_score:
                best_score = score
                best_alpha = alpha

        self.alpha_ = best_alpha

        self.clf_ = LogisticRegression(max_iter=2000, C=self.C)
        self.fitted_cue_ = cue
        self.fitted_content_ = content

        fused_final = np.hstack([
            self.alpha_ * cue,
            (1 - self.alpha_) * content
        ])

        self.clf_.fit(fused_final, y)
        return self

    def predict(self, X):
        X_text = np.array(X)

        cue = _extract_cue_features(X_text)
        content = self.vectorizer.transform(X_text).toarray()

        fused = np.hstack([
            self.alpha_ * cue,
            (1 - self.alpha_) * content
        ])

        return self.clf_.predict(fused)

    def predict_proba(self, X):
        X_text = np.array(X)

        cue = _extract_cue_features(X_text)
        content = self.vectorizer.transform(X_text).toarray()

        fused = np.hstack([
            self.alpha_ * cue,
            (1 - self.alpha_) * content
        ])

        return self.clf_.predict_proba(fused)