currently prototype.py:

with pythia 70m:
64 tok:  Spearman=0.2153  mean_ratio=-0.066  ratio_range=[-9.496, 9.443]

with gpt2:
64 tok:  Spearman=0.9131  mean_ratio=inf
128 tok:  Spearman=0.9342  mean_ratio=0.747
256 tok:  Spearman=0.8740  mean_ratio=inf
512 tok:  Spearman=0.8855  mean_ratio=inf

with gpt2 eps_root=1e-4 instead of 1e-2
64 tok:  Spearman=0.6692  mean_ratio=inf