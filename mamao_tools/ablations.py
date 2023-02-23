import numpy as np
import matplotlib.pyplot as plt
import platform
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
from matplotlib import rc
rc('font', **{'family':'serif', 'serif':['Times']})

PAPER_VERSION = True

tasks = ['fridges', 'counter-to-storage', 'counter-to-pot', 'counter-to-sink']  ##
X = ['PIGINet', 'no value', 'no init', 'no image', 'no image\nno value']  ## , 'no image\nno init'
vals = [
    [0.916,    0.9,     0.741,      0.881,      0.774],  ## ,    0
    [0.9212,   0.9116,  0.8944,     0.5922,     0.4575],  ## , 0.4882
    [0.934,    0.9255,  0.9028,     0.6494,     0.6872],  ## , 0.6391
    [0.914,    0.82,       0.86,       0.7541,     0.71],
]

colors = ['#3498db', '#e74c3c', '#f1c40f', '#1abc9c']  ##

figsize = (6, 4) if not PAPER_VERSION else (6, 3)
if platform.system() == 'Darwin':
    figsize = (5, 2.5)
fig = plt.figure(figsize=figsize)
if not PAPER_VERSION:
    plt.title('Ablation studies on classification accuracy', fontsize=16, pad=35)
else:
    plt.title(' ', fontsize=16, pad=5)
# plt.axis('off')


width = 0.6
n = len(vals)
_X = np.arange(len(X))
for i in range(n):
    plt.bar(_X - width / 2. + i / float(n) * width, vals[i],
            width=width / float(n), align="edge", color=colors[i], alpha=0.6)
plt.xticks(_X, X)
plt.ylabel('Validation accuracy', fontsize=12)
plt.ylim([0.4, 1.0])
plt.legend(tasks, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.5))
plt.tight_layout()

if PAPER_VERSION:
    if platform.system() == 'Darwin':
        plt.savefig('ablations.pdf', bbox_inches='tight')
    else:
        plt.savefig('/home/yang/ablations.pdf', bbox_inches='tight')
else:
    plt.show()