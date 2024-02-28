import numpy as np
import matplotlib.pyplot as plt
f = open("Depth_12.dat","rt")

lines = f.readlines()
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, 56)]
print(lines)

# fig, ax = plt.subplots()
plt.figure()
colors = ['red', 'black', 'blue', 'brown', 'green','yellow','magenta','grey','cyan']
depth_along_years = []
years = []
distances_along_years = []


number =0
s = ''.join(lines)
years_list = s.split('}}')     # splitting along years
for y in years_list:

    if y == '\n':
        break
    year_and_depths = y.split('{{')
    year = year_and_depths[0].strip()
    years.append(year)
    depths = year_and_depths[1:]

    dists_and_depths = depths[0].split(',')
    odds  = dists_and_depths[1::2]
    evens = dists_and_depths[::2]
    odds_cleared = [s.replace('}', '').replace('[', '').replace('{', '').replace('\n', '') for s in odds]
    evens_cleared = [s.replace('}', '').replace('{', '').replace('\n', '') for s in evens]
    odds_numbers = [float(s.strip()) for s in odds_cleared]
    evens_numbers  = [float(s.strip()) for s in evens_cleared]
    depth_along_years.append(odds_numbers)
    distances_along_years.append(evens_numbers)
    plt.plot(np.array(evens_numbers), np.array(odds_numbers), label=str(year), color=colors[number % len(colors)])
    plt.show()
    plt.legend(loc='best')
    number = number + 1
    qq = 0

plt.xlabel('cross-shore distance')
plt.ylabel('depth')
plt.title('Coast profile')

cmap = plt.get_cmap('gnuplot')

qq = 0
dep = np.array(depth_along_years)
dis = np.array(distances_along_years)
qq = 0


lens = np.array([len(s) for s in dis])
observations = lens.min()
depths_uniform    = np.zeros((len(years),observations))
distances_uniform = np.zeros((len(years),observations))
for i in range(len(years)):
    for j in range(observations):
        depths_uniform[i][j]    = dep[i][j]
        distances_uniform[i][j] = dis[i][j]

num_years = [float(y) for y in years]
X,Y = np.meshgrid(num_years,distances_uniform[0])

np.savetxt('num_years.txt',num_years,fmt='%15.5e',delimiter='\n')
np.savetxt('dist_unif.txt',distances_uniform[0],fmt='%15.5e',delimiter='\n')

from matplotlib import cm
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# X,Y = np.meshgrid(num_years,distances_uniform[0])

surf = ax.plot_surface(X, Y, depths_uniform.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('YEAR')
ax.set_ylabel('DISTANCE')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('year_distance.png')
np.savetxt('dep.txt',depths_uniform,fmt = '%15.5e',delimiter='\n')

from fiiter import fit2D
from polynoms import table2dataframe
x,y,z = table2dataframe(depths_uniform[:3,:3],num_years[:3],distances_uniform[0][:3])
fit2D(x,y,z)


# here to call get_loss_one_year
from get_loss_one_year import get_loss_one_year
import torch
C = torch.randn(distances_uniform[0].shape[0]-4)
D = torch.randn(distances_uniform[0].shape[0]-4)
dx = distances_uniform[1][1]

from Ameland_ARIMA import  depth_ARIMA
# depth_ARIMA(depths_uniform,distances_uniform,num_years)

# loss0 = get_loss_one_year(D, C, torch.from_numpy(depths_uniform[:,0]),
#                                 torch.from_numpy(depths_uniform[:,1]), 1.0, dx)
#
# from loss import loss_function
# loss0 = loss_function(depths_uniform,dx,D,C)
from minimizer import minimize
# minimize(D,C,depths_uniform,dx)

plt.figure()
#C,D = get_polynom_coefs()
plt.plot(C.detach().numpy())
np.savetxt('C.txt',C.detach().numpy(),fmt='%15.5e',delimiter='\n')
plt.figure()
plt.plot(D.detach().numpy())
np.savetxt('D.txt',D.detach().numpy(),fmt='%15.5e',delimiter='\n')

qq = 0

plt.figure()
plt.plot()


    #nums = [s.strip() for s in dists_and_depths]