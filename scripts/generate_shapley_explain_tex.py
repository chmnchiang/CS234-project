import sys
import numpy as np

template = r'''
\begin{{tikzpicture}}[
    ar/.style={{very thick, -latex}},
  ]
  \fill[red!30] (0, 0) rectangle (1, 1);
  \fill[red!30] (3, 1) rectangle (4, 3);
  \fill[red!30] (1, 2) rectangle (2, 3);
  \fill[green!30] (3, 0) rectangle (4, 1);
  \draw[thick] (0, 0) grid (4, 4);

  {content}

  {query}
\end{{tikzpicture}}
'''

Qi, Qd = (int(x) for x in sys.argv[1:])

D = [
    (-1, 0),
    (0, -1),
    (1, 0),
    (0, 1),
]
D = [0.5*np.array(x) for x in D]

EPS = 1e-4
data = []

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    i, d, v = line.split()
    i, d, v = int(i), int(d), float(v)

    if v <= EPS:
        continue

    data.append((i, d, v))

val_max = max(v for _, _, v in data)
content = []

for i, d, v in data:
    x = i % 4 + 0.5
    y = 4 - i // 4 - 0.5

    mid = np.array((x, y))
    end = mid + D[d]
    green = 'green!50!black'
    red = 'red'
    out = f'\draw[ar, {green if v > 0 else red}, line width={abs(v) * 2.5 / val_max:.2f}pt] ({x:.2f}, {y:.2f}) -- ({end[0]:.2f}, {end[1]:.2f});'

    content.append(out)

content = '\n  '.join(content)
qx = Qi % 4 + 0.5
qy = 4 - Qi // 4 - 0.5
qd = D[Qd]
query = '\draw[ar] ({}, {}) -- ({}, {});'.format(qx, qy, qx+qd[0], qy+qd[1])
print(template.format(content=content, query=query))
