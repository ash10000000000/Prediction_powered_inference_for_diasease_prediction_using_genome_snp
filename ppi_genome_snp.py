import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from scipy.stats import norm

# PPI and classical CI functions
def ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=0.05):
    n, m = len(Y), len(Yhat_unlabeled)
    mu_labeled = np.mean(Y - Yhat)
    mu_unlabeled = np.mean(Yhat_unlabeled)
    correction = mu_unlabeled
    mu_ppi = mu_labeled + correction
    var_labeled = np.var(Y - Yhat, ddof=1) / n
    var_unlabeled = np.var(Yhat_unlabeled, ddof=1) / m
    var_ppi = var_labeled + var_unlabeled
    z = norm.ppf(1 - alpha / 2)
    return mu_ppi - z * np.sqrt(var_ppi), mu_ppi + z * np.sqrt(var_ppi)

def classical_mean_ci(Y, alpha=0.05):
    n = len(Y)
    mu = np.mean(Y)
    var = np.var(Y, ddof=1) / n
    z = norm.ppf(1 - alpha / 2)
    return mu - z * np.sqrt(var), mu + z * np.sqrt(var)

# Set parameters
true_beta = 2.0
n_train = 10000
n_total = 100000
n_labeled = 50  # Increase to 100 if coverage is low
num_trials = 100
alpha = 0.05
p_snp = [0.5, 0.4, 0.1]

# Generate training data
np.random.seed(0)
snp_train = np.random.choice([0, 1, 2], size=n_train, p=p_snp)
probs_train = 1 / (1 + np.exp(-(-1 + true_beta * snp_train)))
disease_train = np.random.binomial(1, probs_train)
X_train = pd.DataFrame({'SNP': snp_train})
y_train = disease_train

# Train model with calibration
base_model = LogisticRegression(random_state=42)
model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
model.fit(X_train, y_train)

# Generate inference data
np.random.seed(42)
snp = np.random.choice([0, 1, 2], size=n_total, p=p_snp)
probs = 1 / (1 + np.exp(-(-1 + true_beta * snp)))
disease = np.random.binomial(1, probs)
data = pd.DataFrame({'SNP': snp, 'Disease': disease})
true_mean = np.mean(probs)  # True expected probability

# Run trials
results = []
for trial in range(num_trials):
    rng = np.random.RandomState(trial)  # Fixed seed per trial for stability
    rand_idx = rng.permutation(n_total)
    labeled_idx = rand_idx[:n_labeled]
    unlabeled_idx = rand_idx[n_labeled:]
    X_labeled = pd.DataFrame({'SNP': data['SNP'].iloc[labeled_idx]})
    y_labeled = data['Disease'].iloc[labeled_idx].values
    # Skip if unbalanced
    if np.sum(y_labeled) < 2 or np.sum(y_labeled) > len(y_labeled) - 2:
        print(f"Skipping trial {trial}: insufficient positive or negative samples")
        continue
    X_unlabeled = pd.DataFrame({'SNP': data['SNP'].iloc[unlabeled_idx]})
    yhat_labeled = model.predict_proba(X_labeled)[:, 1]
    yhat_unlabeled = model.predict_proba(X_unlabeled)[:, 1]
    ppi_ci = ppi_mean_ci(y_labeled, yhat_labeled, yhat_unlabeled, alpha)
    classical_ci = classical_mean_ci(y_labeled, alpha)
    results.append({'method': 'PPI', 'n': n_labeled, 'lower': ppi_ci[0], 'upper': ppi_ci[1], 'trial': trial})
    results.append({'method': 'Classical', 'n': n_labeled, 'lower': classical_ci[0], 'upper': classical_ci[1], 'trial': trial})

# Imputation
yhat_all = model.predict_proba(pd.DataFrame({'SNP': data['SNP']}))[:, 1]
imputed_ci = classical_mean_ci(yhat_all, alpha)
results.append({'method': 'Imputation', 'n': np.nan, 'lower': imputed_ci[0], 'upper': imputed_ci[1], 'trial': 0})

df = pd.DataFrame(results)
df['width'] = df['upper'] - df['lower']
df['covers'] = (df['lower'] <= true_mean) & (df['upper'] >= true_mean)

# Print average CI widths and coverage
print(f"PPI CI Width: {df[df['method'] == 'PPI']['width'].mean():.3f}")
print(f"Classical CI Width: {df[df['method'] == 'Classical']['width'].mean():.3f}")
print(f"Imputation CI Width: {df[df['method'] == 'Imputation']['width'].mean():.3f}")
print(f"True Mean: {true_mean:.3f}")
print(f"PPI Coverage: {df[df['method'] == 'PPI']['covers'].mean():.3f}")
print(f"Classical Coverage: {df[df['method'] == 'Classical']['covers'].mean():.3f}")
print("Note: Imputation CI is unrealistically narrow as it ignores model uncertainty.")

# 1. Bar Chart with Error Bars
plt.figure(figsize=(8, 5))
methods = ['PPI', 'Classical', 'Imputation']
means = [df[df['method'] == m]['width'].mean() for m in methods]
stds = [df[df['method'] == m]['width'].std() / np.sqrt(num_trials) for m in methods]
plt.bar(methods, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
plt.ylabel('Average CI Width')
plt.title('Average Confidence Interval Width by Method')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig('ci_width_bar.png')
plt.close()

# 2. Box Plot of CI Widths
plt.figure(figsize=(8, 5))
df[df['method'].isin(['PPI', 'Classical'])].boxplot(column='width', by='method', grid=False)
plt.title('Distribution of CI Widths by Method')
plt.suptitle('')
plt.xlabel('Method')
plt.ylabel('CI Width')
plt.savefig('ci_width_boxplot.png')
plt.close()

# 3. Coverage Bar Plot
plt.figure(figsize=(8, 5))
coverage = [df[df['method'] == m]['covers'].mean() for m in ['PPI', 'Classical']]
plt.bar(['PPI', 'Classical'], coverage, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
plt.axhline(1 - alpha, color='black', linestyle='--', label=f'Nominal Coverage (1 - α = {1 - alpha})')
plt.ylabel('Coverage Probability')
plt.title('Coverage of True Mean by Method')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig('coverage_bar.png')
plt.close()
