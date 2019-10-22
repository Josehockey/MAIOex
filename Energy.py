import numpy as npy
import math as math
import matplotlib.pyplot as plt

# Solve linear shallow water equations in radial coordinate on a stretched grid, assuming axisymmetry

DirecOutputEnergy = "/home/giovanni/SOAC/main/Energy/"

###############################  INPUT PARAMETERS

#-----> Arts
Runnumber = 12 # number of the Run                                              
ntplot = 1800 # frequency of output in steps (-> Look at dt for conversion)

#-----> Pysical
g = 9.81
f_cor = 0.0001   
eps = 0.9       # =ro2/ro1
ro1 = 1.0
ro2 = eps * ro1
Href = 5000.0   # reference depth fluid [m]
nlayer_len = 2

############################### GRIDS

#------> Time
dt = 60/3600.0  # time step in seconds
end = 24 # 24 total time of integration                                  
time = npy.arange(0, end, dt)           

#-----> Space
nx_len = 125 #number of points
dx = 1 #grid risolution							      #---> Keep dx fixed and moves the others parameters. 
labda = 0.05 # parameter stretched grid                                       ###SET UP : l=0.05 b = 8000.0 nx_len = 125 perfect for focus 4000km 
b = 20000.0 # parameter stretched grid                                         ###then increase lambda and decrease b to increase the focus power and vice versa

r = npy.zeros(nx_len)   # radial distance grid points u and v                  
c = npy.zeros(nx_len)                                                          
rm = npy.zeros(nx_len)  # radial distance grid points h 
cm = npy.zeros(nx_len)

for nx in range(0, nx_len):
   r[nx] = b * (math.exp(labda*nx) - 1)
   rm[nx] = b * (math.exp(labda*(nx+0.5)) - 1)  
   c[nx] = math.pow(labda*(r[nx]+b),-1)
   cm[nx] = math.pow(labda*(rm[nx]+b),-1)

#----------------------------------------NORMAL

start = input('Energy density domain -> Start : ')
end = input('Energy density domain -> End : ')
result1 = npy.where(r < end)
result2 = npy.where(r > start)
result = npy.arange(npy.amin(result2), npy.amax(result1), 1)

h1 = npy.genfromtxt('h1.txt', delimiter=',')
h2 = npy.genfromtxt('h2.txt', delimiter=',')
u1 = npy.genfromtxt('u1.txt', delimiter=',')
u2 = npy.genfromtxt('u2.txt', delimiter=',')
v1 = npy.genfromtxt('v1.txt', delimiter=',')
v2 = npy.genfromtxt('v2.txt', delimiter=',')

PE = npy.zeros(len(time))
KE = npy.zeros(len(time))
PE_ref = 0
KE_ref = 0
print(rm[int(nx_len*0.9)])
for i in range(0,len(time)) : 
   SUM1 = 0;
   SUM2 = 0;
   for t in range (result[0], result[-1]) :
      if i==0 :
         PE_ref = PE_ref + (h1[i,t]*h1[i,t] + eps*h2[i,t]*h2[i,t] + 2*eps*h1[i,t]*h2[i,t])*rm[t]*math.pow(cm[t],-1)
         KE_ref = KE_ref + ( h1[i,t]*( u1[i,t]*u1[i,t]+v1[i,t]*v1[i,t] ) + eps*h2[i,t]*( u2[i,t]*u2[i,t]+v2[i,t]*v2[i,t] ) )*r[t]*math.pow(c[t],-1)
      SUM1 = SUM1 + (h1[i,t]*h1[i,t] + eps*h2[i,t]*h2[i,t] + 2*eps*h1[i,t]*h2[i,t])*rm[t]*math.pow(cm[t],-1)
      SUM2 = SUM2 + ( h1[i,t]*( u1[i,t]*u1[i,t]+v1[i,t]*v1[i,t] ) + eps*h2[i,t]*( u2[i,t]*u2[i,t]+v2[i,t]*v2[i,t] ) )*r[t]*math.pow(c[t],-1)
   PE[i] = math.pi*g*ro1*SUM1
   KE[i] = math.pi*g*ro1*SUM2

for i in range(0,len(time)) : 
   PE[i] = (PE[i] - math.pi*g*ro1*PE_ref)*10**(-1)  #THE LAST IS NORMALIZATION
   KE[i] = KE[i] - math.pi*g*ro1*KE_ref

plt.figure(figsize=(8,6))
plt.plot(time[:], KE[:],linewidth=3.0, color='red')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('Energy density [J/m^3]',fontsize=14) 
plt.savefig(DirecOutputEnergy+"KE-run"+str(Runnumber)+".png")
plt.show()
plt.close()

plt.figure(figsize=(8,6))
plt.plot(time[:], PE[:],linewidth=3.0, color='black')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('Energy density [J/m^3]',fontsize=14) 
plt.savefig(DirecOutputEnergy+"PE-run"+str(Runnumber)+".png")
plt.show()
plt.close()

plt.figure(figsize=(8,6))
plt.plot(time[:], KE[:]-PE[:],linewidth=3.0, color='blue')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('Energy density [J/m^3]',fontsize=14) 
plt.savefig(DirecOutputEnergy+"TOT-run"+str(Runnumber)+".png")
plt.show()
plt.close()
