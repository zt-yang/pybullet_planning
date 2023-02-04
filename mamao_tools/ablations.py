import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
from matplotlib import rc
rc('font', **{'family':'serif', 'serif':['Times']})

PAPER_VERSION = True

tasks = ['counter-to-storage', 'counter-to-pot']  ## , 'pot-to-storage'
X = ['PIGINet', 'no value', 'no init', 'no image', 'no image\nno value', 'no image\nno init']
vals = [
    [0.9212,    0.9116, 0.8944,     0.5922,     0.4575, 0.4882],
    [0.934,     0.9255, 0.9028,     0.6494,     0.6872, 0.6391],
    # [0.8682,    0.86,   0,          0.674,      0,      0],
]

colors = ['#3498db', '#2ecc71']  ## , '#e74c3c'

figsize = (6, 4) if not PAPER_VERSION else (6, 3)
fig = plt.figure(figsize=figsize)
if not PAPER_VERSION:
    plt.title('Ablation studies on classification accuracy', fontsize=16, pad=35)
else:
    plt.title(' ', fontsize=16, pad=5)
# plt.axis('off')


width = 0.8
n = len(vals)
_X = np.arange(len(X))
for i in range(n):
    plt.bar(_X - width / 2. + i / float(n) * width, vals[i],
            width=width / float(n), align="edge", color=colors[i], alpha=0.6)
plt.xticks(_X, X)
plt.ylim([0, 1.0])
plt.legend(tasks, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2))
plt.tight_layout()

if PAPER_VERSION:
    plt.savefig('/home/yang/ablations.pdf', bbox_inches='tight')
else:
    plt.show()