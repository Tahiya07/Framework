import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# -------------------------
# GRADIENT REVERSAL LAYER
# -------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# -------------------------
# MODEL
# -------------------------
class DomainAdversarialBloomClassifier(nn.Module):

    def __init__(self, feature_dim, num_labels=6):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        # Bloom classifier head
        self.bloom_head = nn.Linear(128, num_labels)

        # Domain classifier head (adversarial)
        self.domain_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, lambd=1.0):
        z = self.encoder(x)

        # Bloom prediction
        bloom_logits = self.bloom_head(z)

        # Domain adversarial branch
        rev_z = grad_reverse(z, lambd)
        domain_logits = self.domain_head(rev_z)

        return bloom_logits, domain_logits


# -------------------------
# WRAPPER CLASS
# -------------------------
class DABCWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, epochs=10, lr=1e-3, lambd=0.5):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.epochs = epochs
        self.lr = lr
        self.lambd = lambd

    def fit(self, X, y, domain_labels):

        X_vec = self.vectorizer.fit_transform(X).toarray()
        self.classes_ = np.array(sorted(set(y)))

        self.model = DomainAdversarialBloomClassifier(
            feature_dim=X_vec.shape[1],
            num_labels=len(self.classes_)
        )

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        y_map = {v: i for i, v in enumerate(self.classes_)}
        self._inverse_y_map = {i: v for v, i in y_map.items()}
        d_map = {v: i for i, v in enumerate(sorted(set(domain_labels)))}

        X_t = torch.tensor(X_vec, dtype=torch.float32)
        y_t = torch.tensor([y_map[i] for i in y], dtype=torch.long)
        d_t = torch.tensor([d_map[i] for i in domain_labels], dtype=torch.long)

        for _ in range(self.epochs):

            bloom_logits, domain_logits = self.model(X_t, self.lambd)

            loss_bloom = F.cross_entropy(bloom_logits, y_t)
            loss_domain = F.cross_entropy(domain_logits, d_t)

            loss = loss_bloom - self.lambd * loss_domain

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self

    def predict(self, X):
        X_vec = self.vectorizer.transform(X).toarray()
        X_t = torch.tensor(X_vec, dtype=torch.float32)

        with torch.no_grad():
            bloom_logits, _ = self.model(X_t, self.lambd)
            pred_idx = torch.argmax(bloom_logits, dim=1).numpy()
        return np.array([self._inverse_y_map[int(i)] for i in pred_idx])

    def predict_proba(self, X):
        X_vec = self.vectorizer.transform(X).toarray()
        X_t = torch.tensor(X_vec, dtype=torch.float32)

        with torch.no_grad():
            bloom_logits, _ = self.model(X_t, self.lambd)
            probs = F.softmax(bloom_logits, dim=1).cpu().numpy()
        return probs
