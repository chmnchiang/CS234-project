S = list('''
...G.
.....
.B...
.G...
.....
'''.strip().split('\n'))
A = 2


template=\
r'''
\begin{{tikzpicture}}[scale=0.45]
  \draw[thick] (0, 0) grid ({n}, {n});
{s}
\end{{tikzpicture}}
'''

templateG = r'''  \draw[fill=green!70!black] ({x1}, {y1}) rectangle ({x2}, {y2});'''
templateB = r'''  \draw[fill=blue] ({x}, {y}) circle (0.35);
  \draw[-latex, line width=0.4mm, red] ({ax1}, {ay1}) -- ({ax2}, {ay2});'''

n = len(S)
ins = []
d = [None, (0, 1), (0, -1), (-1, 0), (1, 0)]

for i in range(n):
    for j in range(n):
        c = S[i][j]
        x = j + 0.5
        y = n - 0.5 - i
        if c == 'G':
            x1, x2 = x-0.3, x+0.3
            y1, y2 = y-0.3, y+0.3
            ins.append(templateG.format(x1=x1, x2=x2, y1=y1, y2=y2))
        elif c == 'B':
            dx, dy = d[A]
            ax1, ay1 = x+dx*0.15, y+dy*0.15
            ax2, ay2 = x+dx*0.9, y+dy*0.9
            ins.append(templateB.format(x=x, y=y, ax1=ax1, ax2=ax2, ay1=ay1, ay2=ay2))

final = template.format(n=n, s='\n'.join(ins))
print(final)
