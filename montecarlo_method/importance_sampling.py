import numpy as np

x = np.array([1,2,3])
pi = np.array([0.1,0.1,0.8])
b = np.array([1/3,1/3,1/3])

e = np.sum(x*pi)
print(f"e_pi[x] = {e}")

n = 100
samples = []

for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)
    mean = np.mean(samples)
    var = np.var(samples)

    print(f"몬테카를로법: {mean} 분산 : {var}")

for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p= b)
    s = x[i]
    rho = pi[i]/b[i]
    samples.append(rho*s)

mean = np.mean(samples)
var = np.var(samples)

print(f"중요도 샘플링 : {mean} 분산 : {var}")