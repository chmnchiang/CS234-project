import sys
import os
import numpy as np

dr = sys.argv[1]
B, W, A = [], [], []

for filename in os.listdir(dr):
    with open(os.path.join(dr, filename)) as f:
        for line in f:
            if line.startswith('best ='):
                best = float(line.split('=')[-1].strip())

            if line.startswith('worst ='):
                worst = float(line.split('=')[-1].strip())

            if line.startswith('average ='):
                avg = float(line.split('=')[-1].strip())
                
        B.append(best - avg)
        W.append(worst - avg)

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
    pts = ' '.join(f'(0, {y:.3f})' for y in dt)
    s = template.format(
        median=median,
        q1=q1,
        q3=q3,
        mx=mx,
        mn=mn,
        pts=pts,
    )
    print(s)

prt(B)
prt(W)
