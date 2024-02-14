import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

PAPER_VERSION = True
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times']})

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

models = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3', 'baseline'] ## 'm4',
colors = ['#3498db', '#3498db', '#3498db',
          '#e74c3c', '#e74c3c', '#e74c3c',
          '#2ecc71']
color_types = ['#3498db', '#e74c3c', '#2ecc71']
training = [0.924, 0.882, 0.893, 0.881, 0.851, 0.886, 0.916] ## 0.886,
testing = [0.89, 0.902, 0.868, 0.898, 0.894, 0.871, 0.916] ## 0.891,

fig, ax = plt.subplots(figsize=(4.5, 4.5))
line = mlines.Line2D([0, 1], [0, 1], color='white')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.scatter(training, testing, c=colors, s=160, alpha=0.8)
ax.set_xlim([0.7, 1])
ax.set_ylim([0.7, 1])
ax.set_xlabel('Accuracy with seen assets', fontsize=18)
ax.set_ylabel('Accuracy with unseen assets', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.locator_params(nbins=4)

groups = ['Trained w/o one food asset', 'Trained w/o one fridge asset', 'Trained on all assets']
handles = [plt.Rectangle((0.5, 0.5), 0.5, 0.5, color=color_types[i]) for i in range(len(groups))]
fig.legend(handles, groups, ncol=1, fontsize=16,
                       loc='lower right', bbox_to_anchor=(0.96, 0.13))
plt.tight_layout()

if PAPER_VERSION:
    if platform.system() == 'Darwin':
        plt.savefig('geometry.pdf', bbox_inches='tight')
    else:
        plt.savefig('/home/yang/geometry.pdf', bbox_inches='tight')
    # plt.savefig('/home/yang/geometry.png', bbox_inches='tight')
else:
    plt.show()