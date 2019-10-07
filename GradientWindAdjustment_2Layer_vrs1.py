import numpy as npy
import math as math
import matplotlib.pyplot as plt

# Solve linear shallow water equations in radial coordinate on a stretched grid, assuming axisymmetry

DirecOutputh = "/Users/Delden/Desktop/GradientWindAdjustment/Results/h/"
DirecOutputv = "/Users/Delden/Desktop/GradientWindAdjustment/Results/v/"
DirecOutputend = "/Users/Delden/Desktop/GradientWindAdjustment/Results/END/"

################################  INPUT PARAMETERS
Runnumber = 13 # number of the Run
dt = 1.0  # time step in seconds
nhours = 18 # 24 total time of integration
nt_len = (nhours * 3600) + 1 # number of time steps
ntplot = 1800 # frequency of output in steps
Href = 5000.0   # reference depth fluid
T_forcing= 6 * 3600.0
a = 500000.0  # horizontal scale of the forcing (mass flux)
Qmax = 0.2 * Href  # amplitude of forcing ("Diabatic mass flux")
RMW = 500000.0  # radius of maximum initial state
r0 = 500000.0  # radius of maximum forcing
v1max_i = 30.0  # max. tang. wind lower layer
v2max_i = 15.0  # max. tang. wind upper layer
################################

dx= 1.0 # grid distance of stretched grid
labda = 0.05 # parameter stretched grid 
b = 20000.0 # parameter stretched grid 

g = 9.81
f_cor = 0.0001
eps = 0.9   # =ro2/ro1
ro1 = 1.0
ro2 = eps * ro1

nx_len = 125    # number of grid points
nlayer_len = 2
Hpert = Qmax

# ARRAYS
r = npy.zeros(nx_len)   # radial distance grid points u and v 
c = npy.zeros(nx_len)
rm = npy.zeros(nx_len)  # radial distance grid points h 
cm = npy.zeros(nx_len)
u = npy.zeros((nt_len,nlayer_len,nx_len))
v = npy.zeros((nt_len,nlayer_len,nx_len))
rv = npy.zeros((nt_len,nlayer_len,nx_len))
zeta = npy.zeros((nt_len,nlayer_len,nx_len))
h = npy.zeros((nt_len,nlayer_len,nx_len))
vgrad = npy.zeros((nlayer_len,nx_len))
M = npy.zeros((nt_len,nlayer_len,nx_len))
geopot = npy.zeros((nt_len,nlayer_len,nx_len))
ps0 = npy.zeros(nt_len)
time = npy.zeros(nt_len)

# Stretched grid
for nx in range(nx_len):
 r[nx] = b * (math.exp(labda*nx) - 1)
 rm[nx] = b * (math.exp(labda*(nx-0.5)) - 1)  # note: rm[0]<0 !!
 c[nx] = math.pow(labda*(r[nx]+b),-1)
 cm[nx] = math.pow(labda*(rm[nx]+b),-1)
  
# Initial condition 
# Prescribe initial tangential velocity: Rankine vortex
for nx in range(nx_len):
 if rm[nx] <= RMW: 
  v[0,0,nx] = v1max_i * rm[nx] / RMW
  v[0,1,nx] = v2max_i * rm[nx] / RMW
  zeta[0,0,nx] = 2 * v1max_i / RMW
  zeta[0,0,nx] = 2 * v2max_i / RMW
  
 if rm[nx] > RMW: 
  v[0,0,nx] = v1max_i * RMW / rm[nx]
  v[0,1,nx] = v2max_i * RMW / rm[nx]
  zeta[0,0,nx] = 0.0
  zeta[0,0,nx] = 0.0

for nlayer in range(nlayer_len):
 for nx in range(1,nx_len-1):
  rv[0,nlayer,nx] = rm[nx] * v[0,nlayer,nx]
  
for nlayer in range(nlayer_len):
 for nx in range(nx_len):
  u[0,nlayer,nx] = 0

# assume gradient wind balance
h[0,0,0] = Href
h[0,1,0] = Href
for nx in range(1,nx_len):
 dphi1 = (((v[0,0,nx]*v[0,0,nx])/rm[nx]) + (f_cor*v[0,0,nx])) * labda * (rm[nx]+b)
 dphi2 = (((v[0,1,nx]*v[0,1,nx])/rm[nx]) + (f_cor*v[0,1,nx])) * labda * (rm[nx]+b) 
 dh2 = (1/((1-eps)*g)) * (dphi2 - dphi1)
 dh1 = (dphi1/g) - (eps*dh2)
 h[0,0,nx] = h[0,0,nx-1]+dh1
 h[0,1,nx] = h[0,1,nx-1]+dh2

for nx in range(nx_len):
 geopot[0,0,nx] = g * (h[0,0,nx] + (eps * h[0,1,nx] ))
 geopot[0,1,nx] = g * (h[0,1,nx] + h[0,1,nx] )

ps0ref =  g * ((ro1 * Href) + (ro2 * Href)) # central surface pressure : reference state
ps0[0] = g * ((ro1 * h[0,0,0]) + (ro2 * h[0,1,0])) # central surface pressure : initial state

# Specify forcing (mass flux)
for nt in range(nt_len):
 time[nt] = nt * dt
 for nx in range(nx_len):
  if time[nt]<=T_forcing: 
   M[nt,0,nx] = -(Qmax/T_forcing) * math.exp(-math.pow((r[nx]-r0)/a,2))   # layer 1
   M[nt,1,nx] = (Qmax/(eps*T_forcing)) * math.exp(-math.pow((r[nx]-r0)/a,2))   # layer 1
  if time[nt]>T_forcing: 
   M[nt,0,nx] = 0.0 
   M[nt,1,nx] = 0.0
    
#########################################################################################
# BEGIN TIME LOOP

for nt in range(1,nt_len):
 
 time[nt] = time[nt] / 3600.0  # time in hours
  
 for nlayer in range(nlayer_len):
  for nx in range(2,nx_len-1):
   h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * (Href / r[nx]) * ( c[nx]*((rm[nx+1]*u[nt-1,nlayer,nx+1])-(rm[nx]*u[nt-1,nlayer,nx]))) )   # h on r-grid
   h[nt,nlayer,nx] = h[nt,nlayer,nx] + (dt * M[nt,nlayer,nx])     
# Boundary conditions
  h[nt,nlayer,1] = h[nt,nlayer,2]
  h[nt,nlayer,0] = h[nt,nlayer,1]
  h[nt,nlayer,nx_len-1] = h[nt,nlayer,nx_len-2]

# Compute the new geopotential
 for nx in range(nx_len):
  geopot[nt,0,nx] = g * (h[nt,0,nx] + (eps * h[nt,1,nx] ))
  geopot[nt,1,nx] = g * (h[nt,0,nx] + h[nt,1,nx] )
  
# Compute the gradient wind, vgrad at x=1,2,3...   , i.e at the same grid points as geopotential and height
 for nx in range(1,nx_len-1):
  c1 = f_cor * r[nx]
  gradphi0 = c[nx] * (geopot[nt,0,nx+1]-geopot[nt,0,nx-1])/2  # radial gradient geopotential calculated on rm-grid
  gradphi1 = c[nx] * (geopot[nt,1,nx+1]-geopot[nt,1,nx-1])/2
  c2 = ((c1 * c1)+ (4*r[nx]*gradphi0))
  if c2>0: vgrad[0,nx] = - 0.5 * (c1 - math.pow(c2,0.5))
  c2 = ((c1 * c1)+ (4*rm[nx]*gradphi1))
  if c2>0: vgrad[1,nx] = - 0.5 * (c1 - math.pow(c2,0.5))
# Boundary condition
 vgrad[0,0] = 0.0
 vgrad[1,0] = 0.0
# Compute balanced relative vorticity on intermediate points (at x=0.5,1.5,...
 for nx in range(1,nx_len-1):
  zeta[nt,0,nx] = (1/rm[nx]) * cm[nx] * ((rm[nx]*vgrad[0,nx]) - (rm[nx-1]*vgrad[0,nx-1]))
  zeta[nt,1,nx] = (1/rm[nx]) * cm[nx] * ((rm[nx]*vgrad[1,nx]) - (rm[nx-1]*vgrad[1,nx-1]))

# v: forward (predictor) in time using previous value of u
# u: Backward in time using new values of h and v
# v: Backward (corrector) in time using new value of u
 for nlayer in range(nlayer_len):
  for nx in range(1,nx_len-1):
   v[nt,nlayer,nx] = v[nt-1,nlayer,nx] - ( dt * (f_cor + zeta[nt-1,nlayer,nx]) * u[nt-1,nlayer,nx])
   u[nt,nlayer,nx] = u[nt-1,nlayer,nx] - (dt * cm[nx] * (geopot[nt,nlayer,nx]-geopot[nt,nlayer,nx-1]))  # u & v on rm-grid
   u[nt,nlayer,nx] = u[nt,nlayer,nx] + (dt * f_cor * v[nt,nlayer,nx])
   u[nt,nlayer,nx] = u[nt,nlayer,nx] + (dt * v[nt,nlayer,nx] * v[nt,nlayer,nx] / rm[nx])
   v[nt,nlayer,nx] = v[nt-1,nlayer,nx] - ( dt * (f_cor + zeta[nt,nlayer,nx]) * u[nt,nlayer,nx])

# Boundary conditions
 for nlayer in range(nlayer_len):
  u[nt,nlayer,nx_len-1] = u[nt,nlayer,nx_len-2]
  v[nt,nlayer,nx_len-1] = v[nt,nlayer,nx_len-2]
  u[nt,nlayer,0] = 0
  v[nt,nlayer,0] = 0

# Backward (corrector) time step in h
 for nlayer in range(nlayer_len):
  for nx in range(2,nx_len-1):
   h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * (Href / r[nx]) * ( c[nx]*((rm[nx+1]*u[nt,nlayer,nx+1])-(rm[nx]*u[nt,nlayer,nx]))) )   # h on r-grid
   h[nt,nlayer,nx] = h[nt,nlayer,nx] + (dt * M[nt,nlayer,nx])     

# Boundary conditions
 for nlayer in range(nlayer_len):
  #h[nt,nlayer,1] = h[nt,nlayer,2]
  h[nt,nlayer,0] = h[nt,nlayer,1]
  h[nt,nlayer,nx_len-1] = h[nt,nlayer,nx_len-2]

 ps0[nt] = g * ((ro1 * h[nt,0,0]) + (ro2 * h[nt,1,0])) # central surface pressure
 ps0[nt] = (ps0[nt]-ps0ref)/100
 
 
# BEGIN OUTPUT 
 if (nt == math.trunc(nt / ntplot) * ntplot):
    
  #PLOT h	
  plt.figure(figsize=(8,6))
  if Hpert<0: plt.axis([0,4000,Href+(2*Hpert),Href-(2*Hpert)])
  if Hpert>0: plt.axis([0,4000,Href-(2*Hpert),Href+(2*Hpert)])
  plt.plot(r/1000, h[nt,0,:],linewidth=3.0, color='red')
  plt.plot(r/1000, h[nt,1,:],linewidth=3.0, color='blue')
  plt.xlabel('radius [km]',fontsize=14)
  plt.ylabel('h [m]',fontsize=14)
  plt.text(100,Href-(1.5*Hpert),"t="+str(nt*dt/3600)+" hours")
  plt.text(100,Href-(1.65*Hpert),"red: lower layer",color='red')
  plt.text(100,Href-(1.8*Hpert),"blue: upper layer",color='blue')
  #plt.savefig(DirecOutputh+"GradWindAdjustment-Run"+str(Runnumber)+"-h"+str(nt)+".png")
  #plt.show()
  plt.close()
  
  #PLOT v
  plt.figure(figsize=(8,6))
  
  
  if v1max_i>=0 and v1max_i<5.0: 
   plt.axis([0,4000,-5,5])
   plt.text(100,4.5,"t="+str(nt*dt/3600)+" hours")
   plt.text(100,4.0,"red: lower layer",color='red')
   plt.text(100,3.6,"blue: upper layer",color='blue')

  if v1max_i>=5.0: 
   plt.axis([0,4000,-2*v1max_i,2*v1max_i])
   plt.text(100,0.9*2*v1max_i,"t="+str(nt*dt/3600)+" hours") 
   plt.text(100,0.8*2*v1max_i,"red: lower layer",color='red')
   plt.text(100,0.72*2*v1max_i,"blue: upper layer",color='blue')

  plt.plot(rm/1000, v[nt,0,:],linewidth=3.0, color='red')
  plt.plot(rm/1000, v[nt,1,:],linewidth=3.0, color='blue')
  plt.xlabel('radius [km]',fontsize=14)
  plt.ylabel('v [m/s]',fontsize=14)
  #plt.savefig(DirecOutputv+"GradWindAdjustment-Run"+str(Runnumber)+"-v"+str(nt)+".png")
  #plt.show()
  plt.close()
# END OUTPUT 

# END TIME LOOP
#########################################################################################

# compute local internal Rossby radius
RR = math.pow((1-eps)*g*Href,0.5)/(f_cor + (2 * v1max_i / RMW)) 

nt = nt_len - 1 
print ("number  Dps(r=0)     a  Rossby radius  v1max_i  v2max_i   RMW      Qmax/Href   T_forcing   Time")
print ("%5.0f, %8.2f, %8.2f, %8.2f,%8.2f,%8.2f, %8.2f, %8.2f, %8.2f, %8.2f")% (Runnumber,ps0[nt],a/1000,RR/1000,v1max_i,v2max_i,RMW/1000,Qmax/Href,T_forcing/3600.,nt*dt/3600.)

# Compute the gradient wind, vgrad at the end of the run and plot both vgrad and v
nt = nt_len - 1
for nx in range(1,nx_len-1):
 c1 = f_cor * rm[nx]
 gradphi0 = cm[nx] * ((geopot[nt,0,nx]-geopot[nt,0,nx-1]))   # pressur gradient calculated on rm-grid
 gradphi1 = cm[nx] * ((geopot[nt,1,nx]-geopot[nt,1,nx-1]))
 c2 = ((c1 * c1)+ (4*rm[nx]*gradphi0))
 if c2>0: vgrad[0,nx] = - 0.5 * (c1 - math.pow(c2,0.5))
 c2 = ((c1 * c1)+ (4*rm[nx]*gradphi1))
 if c2>0: vgrad[1,nx] = - 0.5 * (c1 - math.pow(c2,0.5))

# PLOT v and vgrad
plt.figure(figsize=(8,6))

if v1max_i>=0 and v1max_i<5.0: 
 plt.axis([0,4000,-5,5])
 plt.text(100,4.5,"t="+str(nt*dt/3600)+" hours")
 plt.text(100,4.0,"red: lower layer (v:thick; vgrad: thin)",color='red')
 plt.text(100,3.6,"blue: upper layer (v:thick; vgrad: thin)",color='blue')

if v1max_i>=5.0: 
 plt.axis([0,4000,-2*v1max_i,2*v1max_i])
 plt.text(100,0.9*2*v1max_i,"t="+str(nt*dt/3600)+" hours") 
 plt.text(100,0.8*2*v1max_i,"red: lower layer (v:thick; vgrad: thin)",color='red')
 plt.text(100,0.72*2*v1max_i,"blue: upper layer (v:thick; vgrad: thin)",color='blue')
 
plt.plot(rm[:]/1000, v[nt,0,:],linewidth=3.0, color='red')
plt.plot(rm[:]/1000, v[nt,1,:],linewidth=3.0, color='blue')
plt.plot(rm[:]/1000, vgrad[0,:],linewidth=1.0, color='red')
plt.plot(rm[:]/1000, vgrad[1,:],linewidth=1.0, color='blue')

plt.xlabel('radius [km]',fontsize=14)
plt.ylabel('v [m/s]',fontsize=14) 
 
#plt.savefig(DirecOutputend+"GradientWind-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of central surface pressure (surface = lower boundary)
plt.figure(figsize=(8,6))
plt. axis([0,nt_len*dt/3600,-5,5])
plt.plot(time, ps0,linewidth=2.0, color='black')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('surface pressure deficit at r=0 [hPa]',fontsize=14) 

#plt.savefig(DirecOutputend+"ps0-Run"+str(Runnumber)+".png")
plt.show()
plt.close()




 
