import pandas as pd
import matplotlib.pyplot as plt

groups = ['Image', 'Text', 'Relation', 'Value']

df = pd.DataFrame({
    'Area': ['Image', 'Image', 'Image',
             'Text', 'Text',
             'Relation', 'Relation', 'Relation',
             'Value', 'Value'],
    'Rank': ['clip feature', 'one-hot', 'both',
             'clip feature', 'one-hot',
             'all', 'random_drop', 'no_init',
             'no_value', 'no_pose'],
    'Accuracy': [0.873,0.858,0.930,
              70,114,
              110,195,92,
              44,179],
    # 'True Positive': [630,426,312,
    #                   191,374,
    #                   109,194,708,
    #                   236,806],
    # 'True Negative': [630,426,312,
    #                   191,374,
    #                   109,194,708,
    #                   236,806]
})
df = df.set_index(['Area', 'Rank'])

fig = plt.figure(figsize=(12, 4))
plt.title('Ablation studies on classification accuracy', fontsize=16)
plt.axis('off')

for i, l in enumerate(groups):
    if i == 0:
        sub1 = fig.add_subplot(141+i)

    else:
        sub1 = fig.add_subplot(141+i, sharey=sub1)

    df.loc[l].plot(kind='bar', ax=sub1)
    sub1.set_xticklabels(sub1.get_xticklabels(), rotation=0)
    sub1.set_xlabel(l, fontsize=12)
    sub1.tick_params(axis='x', which='major', pad=10)

plt.tight_layout()
plt.show()