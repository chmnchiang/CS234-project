import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

FILE = [0, 1, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

G = []
B = []
O = []

for fn in FILE:
    file_name = f'./result/shapley_identify/seed-{fn}.txt'
    with open(file_name) as f:
        it = iter(f)

        mp = {}

        for i in range(16):
            line = next(it)
            _, i, l = line.split()
            i = int(i)
            l = int(l)
            mp[i] = l


        line = next(it)
        order = eval(line)
        best = [i for i, _ in order[:4]]
        others = [i for i, _ in order[4:-4]]
        worst = [i for i, _ in order[-4:]]
        
        for b in best:
            G.append(mp[b])

        for w in worst:
            B.append(mp[w])

        for o in others:
            O.append(mp[o])

# G.sort()
# B.sort()
# O.sort()

# G = G[5:-5]
# B = B[5:-5]

template = r'''\addplot+[
  thick,
  mark size=2pt,
  mark=x,
  boxplot prepared={{
    median={median},
    upper quartile={q3},
    lower quartile={q1},
    upper whisker={mx},
    lower whisker={mn},
  }},
] coordinates {{
  {pts}
}};'''

def prt(dt):
    dt = np.array(dt)
    median = np.median(dt)
    q1 = np.quantile(dt, 0.25)
    q3 = np.quantile(dt, 0.75)
    iqr = q3 - q1
    lw = q1 - 1.5*iqr
    hi = q3 + 1.5*iqr

    mx = max(x for x in dt if x <= hi)
    mn = min(x for x in dt if x >= lw)
    pts = ' '.join(f'(0, {y})' for y in dt)
    s = template.format(
        median=median,
        q1=q1,
        q3=q3,
        mx=mx,
        mn=mn,
        pts=pts,
    )
    print(s)


X = ['Best' for _ in G] + ['Worst' for _ in B] + ['Others' for _ in O]
Y = G + B + O
dt = {
    'Length': Y,
    'Subset type': X,
}
df = pd.DataFrame(dt)
sns.set(style="whitegrid", font_scale=1.3)
palette = sns.color_palette('muted')
palette = [palette[i] for i in (0, 3, 7)]
sns.violinplot(x='Subset type', y='Length', data=df, palette=palette)
plt.show()
# sns.despine(left=True)
# sns.swarmplot(x=X, y=Y, color='.2')
# sns.boxenplot(x=X, y=Y, palette='muted')
# for g in G:
    # print(1, g)
# for b in B:
    # print(2, b)
# for o in O:
    # print(3, o)
# print(np.mean(G), np.mean(B), np.mean(O))
