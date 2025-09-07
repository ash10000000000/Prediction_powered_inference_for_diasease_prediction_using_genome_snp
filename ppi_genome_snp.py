import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import urllib.request
import subprocess
from tqdm import tqdm
import tarfile

# Toggle for synthetic data if download fails
use_synthetic = False  # Set to True if download takes too long

# Download progress bar
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {url} to {output_path} (~{os.path.getsize(output_path)/1024/1024:.1f} MB)")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    else:
        print(f"{output_path} already exists, skipping download")

# Download and setup bcftools
def setup_bcftools():
    bcftools_url = "https://github.com/samtools/bcftools/releases/download/1.20/bcftools-1.20.tar.bz2"
    bcftools_tar = "bcftools-1.20.tar.bz2"
    bcftools_dir = "bcftools-1.20"
    bcftools_bin = os.path.join(bcftools_dir, "bcftools")
    if not os.path.exists(bcftools_bin):
        print("Downloading and setting up bcftools (~5 MB)")
        download_file(bcftools_url, bcftools_tar)
        with tarfile.open(bcftools_tar, "r:bz2") as tar:
            tar.extractall()
        subprocess.run(["make"], cwd=bcftools_dir, check=True)
        os.chmod(bcftools_bin, 0o755)
    return bcftools_bin

# Download and filter VCF
vcf_url = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
tbi_url = vcf_url + ".tbi"
vcf_file = "ALL.chr22.vcf.gz"
tbi_file = vcf_file + ".tbi"
snp_vcf = "single_snp.vcf"

if not use_synthetic and not os.path.exists(snp_vcf):
    try:
        print("Downloading 1000 Genomes chr22 VCF (~200 MB) and index (~100 KB)")
        download_file(vcf_url, vcf_file)
        download_file(tbi_url, tbi_file)
        bcftools_bin = setup_bcftools()
        print("Filtering VCF for rs738491 (chr22:16050408)")
        subprocess.run([bcftools_bin, "view", "-r", "22:16050408-16050408", vcf_file, "-o", snp_vcf], check=True)
    except Exception as e:
        print(f"Download/filter failed: {e}. Switching to synthetic data.")
        use_synthetic = True

# Simplified PPI mean CI function
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

# Simplified classical mean CI function
def classical_mean_ci(Y, alpha=0.05):
    n = len(Y)
    mu = np.mean(Y)
    var = np.var(Y, ddof=1) / n
    z = norm.ppf(1 - alpha / 2)
    return mu - z * np.sqrt(var), mu + z * np.sqrt(var)

# Load or generate data
if use_synthetic:
    print("Using synthetic data (0 MB download)")
    np.random.seed(42)
    n_samples = 1000
    snp = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.4, 0.1])
    true_beta = 0.5
    probs = 1 / (1 + np.exp(-(-1 + true_beta * snp)))
    disease = np.random.binomial(1, probs)
    data = pd.DataFrame({'SNP': snp, 'Disease': disease})
else:
    print("Parsing real 1000 Genomes VCF data (rs738491)")
    genotypes = []
    with open(snp_vcf, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            fields = line.strip().split('\t')
            gts = fields[9:]
            genotypes.extend([sum(map(int, gt.split('|')[0].split('/'))) for gt in gts if gt != './.'])
    snp = np.array(genotypes[:1000])  # Limit to 1000 samples for speed
    np.random.seed(42)
    probs = 1 / (1 + np.exp(-(-1 + 0.5 * snp)))
    disease = np.random.binomial(1, probs)
    data = pd.DataFrame({'SNP': snp, 'Disease': disease})

# Problem setup
alpha = 0.05
n_total = len(data)
n_labeled = 50
num_trials = 10
results = []

# Run PPI and classical inference
for trial in range(num_trials):
    rand_idx = np.random.permutation(n_total)
    X_labeled = data['SNP'].iloc[rand_idx[:n_labeled]].values.reshape(-1, 1)
    y_labeled = data['Disease'].iloc[rand_idx[:n_labeled]].values
    X_unlabeled = data['SNP'].iloc[rand_idx[n_labeled:]].values.reshape(-1, 1)

    model = LogisticRegression(random_state=42)
    model.fit(data[['SNP']], data['Disease'])
    yhat_labeled = model.predict_proba(X_labeled)[:, 1]
    yhat_unlabeled = model.predict_proba(X_unlabeled)[:, 1]

    ppi_ci = ppi_mean_ci(y_labeled, yhat_labeled, yhat_unlabeled, alpha)
    classical_ci = classical_mean_ci(y_labeled, alpha)

    results.append({'method': 'PPI', 'n': n_labeled, 'lower': ppi_ci[0], 'upper': ppi_ci[1], 'trial': trial})
    results.append({'method': 'Classical', 'n': n_labeled, 'lower': classical_ci[0], 'upper': classical_ci[1], 'trial': trial})

# Imputation CI
imputed_ci = classical_mean_ci(model.predict_proba(data[['SNP']])[:, 1] > 0.5, alpha)
results.append({'method': 'Imputation', 'n': np.nan, 'lower': imputed_ci[0], 'upper': imputed_ci[1], 'trial': 0})

# Create DataFrame
df = pd.DataFrame(results)
df['width'] = df['upper'] - df['lower']

# Plot
plt.figure(figsize=(6, 3))
for method, color in [('PPI', 'blue'), ('Classical', 'orange'), ('Imputation', 'green')]:
    df_method = df[df['method'] == method]
    if method == 'Imputation':
        plt.errorbar([method], [df_method['lower'].mean()], yerr=[[0], [0]], fmt='o', color=color, label=method)
    else:
        for trial in range(5):
            df_trial = df_method[df_method['trial'] == trial]
            plt.errorbar([method], [df_trial['lower'].iloc[0]], 
                         yerr=[[df_trial['lower'].iloc[0] - df_trial['lower'].iloc[0]], 
                               [df_trial['upper'].iloc[0] - df_trial['lower'].iloc[0]]], 
                         fmt='o', color=color, alpha=0.5)
        plt.errorbar([method], [df_method['lower'].mean()], 
                     yerr=[[df_method['lower'].mean() - df_method['lower'].min()], 
                           [df_method['upper'].max() - df_method['lower'].mean()]], 
                     fmt='o', color=color, label=method)
plt.axhline(np.mean(data['Disease']), color='black', linestyle='--', label='True Mean')
plt.ylabel('Disease Probability')
plt.title('PPI vs Classical vs Imputation (1000G SNP)')
plt.legend()
plt.grid(True)
plt.savefig('snp_ppi_ci.png')
plt.show()

# Print average CI widths
print(f"PPI CI Width: {df[df['method'] == 'PPI']['width'].mean():.3f}")
print(f"Classical CI Width: {df[df['method'] == 'Classical']['width'].mean():.3f}")
print(f"Imputation CI Width: {df[df['method'] == 'Imputation']['width'].mean():.3f}")