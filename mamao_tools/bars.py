import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

groups = ['Text', 'Image', 'Relation', 'Value']

models = [
    'baseline', 'rel=one-hot',
    'img=feature', 'img=one-hot',
    'rel=all', 'rel=no_init',
    'val=no-value', 'val=no-pose',
]

df = pd.DataFrame({
    'Area': ['Text', 'Text',
             'Image', 'Image',
             'Relation', 'Relation',
             'Value', 'Value'],
    'Rank': ['baseline', 'one-hot',
             'clip feature', 'one-hot',
             'all', 'no_init',
             'no_value', 'no_pose'],
    'Train Accuracy': [0.873,0.858,
              0.70,0.114,
              0.195,0.92,
              0.44,0.179],
    'Val Accuracy': [0.873,0.858,
              0.70,0.114,
              0.195,0.92,
              0.44,0.179],
    'Val True Positive': [0.873,0.858,
              0.70,0.114,
              0.195,0.92,
              0.44,0.179],
    'Val False Positive': [0.873,0.858,
              0.70,0.114,
              0.195,0.92,
              0.44,0.179],
})
df = df.set_index(['Area', 'Rank'])

fig = plt.figure(figsize=(12, 4))
plt.title('Ablation studies on classification accuracy', fontsize=16, pad=35)
plt.axis('off')

colors = ['b', 'y', 'g', 'r']
colors = ['#3498db', '#f1c40f', '#2ecc71', '#e74c3c']

for i, l in enumerate(groups):
    if i == 0:
        sub1 = fig.add_subplot(141+i)
    else:
        sub1 = fig.add_subplot(141+i, sharey=sub1)

    df.loc[l].plot(kind='bar', ax=sub1, color=colors)
    sub1.set_xticklabels(sub1.get_xticklabels(), rotation=0)
    sub1.set_xlabel(l, fontsize=12)
    sub1.tick_params(axis='x', which='major', pad=10)
    sub1.get_legend().remove()

handles, labels = sub1.get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.9))
plt.tight_layout()
plt.show()