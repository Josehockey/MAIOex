############################### OUTPUT PATH   (1)

import numpy as npy
import math as math
import matplotlib.pyplot as plt

# Solve linear shallow water equations in radial coordinate on a stretched grid, assuming axisymmetry
# Non-linear analysis over the initial background state imposed

DirecOutputh =   "/home/giovanni/SOAC/main2/Nlin/h/"
DirecOutputv =   "/home/giovanni/SOAC/main2/Nlin/v/"
DirecOutputu =   "/home/giovanni/SOAC/main2/Nlin/u/"
DirecOutputend = "/home/giovanni/SOAC/main2/Nlin/END/"

############################### INPUT PARAMETERS  (2)

#-----> Arts
Runnumber = 100      # number of the Run                                              
ntplot = 1800      # frequency of output in steps (-> Look at dt for conversion)

#-----> Pysical
g = 9.81
f_cor = 0.0001     # 45 degree
eps = 0.9      
ro1 = 1.0
ro2 = eps * ro1
Href = 5000.0
nlayer_len = 2
PSOref =  g * ((ro1 * Href) + (ro2 * Href))        # central surface pressure : reference state    

#-----> Forcing
T_forcing= 4 * 3600.0          # time scale forcing [s]
a = 500000.0                   # horizontal scale of the forcing [m]
Qmax = 0.4*Href                # amplitude of forcing ("Diabatic mass flux")       
r0 = 500000.0                  # radius of maximum forcing [m]

#-----> Initial
RMW = 500000.0      # radius of maximum initial state (velocity)
v1max_i = 0.0      # max. tang. wind lower layer                                   
v2max_i = 0.0      # max. tang. wind upper layer

############################### GRIDS (3)

#------> Time
dt = 2            # time step in seconds
end = 48*3600     # 24 total time of integration                                  
time = npy.arange(0, end, dt)           

#-----> Space
nx_len = 125      # number of points
dx = 1            # grid risolution							     . 
labda = 0.05      # parameter stretched grid
b = 20000.0       # parameter stretched grid                                         

r = npy.zeros(nx_len)   # radial distance grid points u and v                  
c = npy.zeros(nx_len)   # dr^-1                                                       
rm = npy.zeros(nx_len)  # radial distance grid points h 
cm = npy.zeros(nx_len)  # drm^-1

for nx in range(0, nx_len):
   r[nx] = b * (math.exp(labda*nx) - 1)
   rm[nx] = b * (math.exp(labda*(nx+0.5)) - 1)  
   c[nx] = math.pow(labda*(r[nx]+b),-1)
   cm[nx] = math.pow(labda*(rm[nx]+b),-1)

#-----> Velocities and algorithm
u = npy.zeros((len(time),nlayer_len,nx_len))
v = npy.zeros((len(time),nlayer_len,nx_len))
h = npy.zeros((len(time),nlayer_len,nx_len))
M = npy.zeros((len(time),nlayer_len,nx_len))
geopot = npy.zeros((len(time),nlayer_len,nx_len))                               

#-----> Variables of interest
vgrad = npy.zeros((nlayer_len,nx_len))       #Final wind grid balance--> Time indipendent 
PSO = npy.zeros(len(time))                   #Timeseries of pressure in the centre
vmax = npy.zeros(len(time))                  #Timeseries of maximum velocity 
PE = npy.zeros(len(time))                    #Timeseries of the kinetic energy
KE = npy.zeros(len(time))                    #Timeseries of potential energy
PE_ref = 0			
KE_ref = 0

############################### INITIAL CONDITIONS (4)

#-----> Rankine vortex
for nx in range(0, nx_len):
   if r[nx] <= RMW: 
      v[0,0,nx] = v1max_i * r[nx] / RMW
      v[0,1,nx] = v2max_i * r[nx] / RMW
  
   if r[nx] > RMW: 
      v[0,0,nx] = v1max_i * RMW / r[nx]
      v[0,1,nx] = v2max_i * RMW / r[nx]
  
for nlayer in range(nlayer_len):
   for nx in range(nx_len):
      u[0,nlayer,nx] = 0		 # Initial radial velocity set to zero.

#-----> Gradient wind balance at t=0
h[0,0,-1] = Href 
h[0,1,-1] = Href
for nx in range(1,nx_len):
   dphi1 = (((v[0,0,-nx]*v[0,0,-nx])/rm[-nx]) + (f_cor*v[0,0,-nx])) * labda * (rm[-nx]+b)        
   dphi2 = (((v[0,1,-nx]*v[0,1,-nx])/rm[-nx]) + (f_cor*v[0,1,-nx])) * labda * (rm[-nx]+b) 
   dh2 = (1/((1-eps)*g)) * (dphi2 - dphi1)
   dh1 = (dphi1/g) - (eps*dh2)
   h[0,0,-nx-1] = h[0,0,-nx]-dh1
   h[0,1,-nx-1] = h[0,1,-nx]-dh2                                                    
                                                                                
for nx in range (nx_len):
   geopot[0,0,nx] = g * (h[0,0,nx] + (eps * h[0,1,nx] ))
   geopot[0,1,nx] = g * (h[0,0,nx] + h[0,1,nx] )                              

#-----> Specify forcing
for nt in range(len(time)):
   for nx in range(nx_len):
      if time[nt]<=T_forcing: 
         M[nt,0,nx] = -(Qmax/T_forcing) * math.exp(-math.pow((r[nx]-r0)/a,2))     	
         M[nt,1,nx] = (Qmax/(eps*T_forcing)) * math.exp(-math.pow((r[nx]-r0)/a,2)) 
      if time[nt]>T_forcing: 
         M[nt,0,nx] = 0.0
         M[nt,1,nx] = 0.0

#-----> Variables of interest: initialization     
PSO[0] = g * ((ro1 * h[0,0,0]) + (ro2 * h[0,1,0])) # central surface pressure : initial state
vmax[0] = npy.amax(v[0,0,:]) 			   # initial maximum velocity

#-----> Energy inspection 
start = 5000000     #starting inspection [m]
end = 10000000      #end inspection [m]
result1 = npy.where(r < end)
result2 = npy.where(r > start)
result = npy.arange(npy.amin(result2), npy.amax(result1), 1)
for nx in range (result[0], result[-1]):
   PE_ref = PE_ref + (h[0,0,nx]*h[0,0,nx] + eps*h[0,1,nx]*h[0,1,nx] + 2*eps*h[0,0,nx]*h[0,1,nx])*rm[nx]*math.pow(cm[nx],-1)
   KE_ref = KE_ref + (h[0,0,nx]*(u[0,0,nx]*u[0,0,nx]+v[0,0,nx]*v[0,0,nx]) + eps*h[0,1,nx]*(u[0,1,nx]*u[0,1,nx]+v[0,1,nx]*v[0,1,nx]))*rm[nx]*math.pow(cm[nx],-1)
  
############################### ALGORITHM (5)

for nt in range(1,len(time)):

   #---> Compute the new height (NON LINEAR VERSION FULL EQUATIONS : FORWARD H)
   for nlayer in range(nlayer_len):
      for nx in range(1,nx_len-1):
         h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * h[nt-1,nlayer,nx] * c[nx]*(u[nt-1,nlayer,nx+1]- u[nt-1,nlayer,nx]) )   
         h[nt,nlayer,nx] = h[nt,nlayer,nx] - u[nt-1,nlayer,nx]*(h[nt-1,nlayer,nx+1] - h[nt-1,nlayer,nx])*cm[nx]*dt
         h[nt,nlayer,nx] = h[nt,nlayer,nx] - h[nt-1,nlayer,nx]*u[nt-1,nlayer,nx]/rm[nx]
         h[nt,nlayer,nx] = h[nt,nlayer,nx] + (dt * M[nt,nlayer,nx])   		                              
      #---> Boundary conditions (ensure constant h at r=0 and r=+infty)
      h[nt,nlayer,0] = h[nt,nlayer,1]
      h[nt,nlayer,-1] = h[nt,nlayer,-2]

   #---> Compute the new geopotential
   for nx in range(nx_len):
      geopot[nt,0,nx] = g * (h[nt,0,nx] + (eps * h[nt,1,nx] ))
      geopot[nt,1,nx] = g * (h[nt,0,nx] + h[nt,1,nx] )

   #---> v: forward (predictor) in time using previous value of u
   #---> u: Backward in time using new values of h and v
   #---> v: Backward (corrector) in time using new value of u
   for nlayer in range(nlayer_len):
      for nx in range(1,nx_len-1):
         v[nt,nlayer,nx] = v[nt-1,nlayer,nx] - ( dt * (f_cor + (v[nt-1,nlayer,nx]/r[nx]) + (v[nt-1,nlayer,nx+1] - v[nt-1,nlayer,nx])*c[nx]) * u[nt-1,nlayer,nx])
         u[nt,nlayer,nx] = u[nt-1,nlayer,nx] - (dt * cm[nx] * (geopot[nt,nlayer,nx]-geopot[nt,nlayer,nx-1]))  
         u[nt,nlayer,nx] = u[nt,nlayer,nx] + (dt * f_cor * v[nt,nlayer,nx])
         u[nt,nlayer,nx] = u[nt,nlayer,nx] + (dt * v[nt,nlayer,nx] * v[nt,nlayer,nx] / r[nx])
         v[nt,nlayer,nx] = v[nt-1,nlayer,nx] - ( dt * (f_cor + (v[nt-1,nlayer,nx]/r[nx]) + (v[nt-1,nlayer,nx+1] - v[nt-1,nlayer,nx])*c[nx]) * u[nt,nlayer,nx])

   #---> Boundary conditions (continuity at the boundaries and zero in the centre)
   for nlayer in range(nlayer_len):
      u[nt,nlayer,-1] = u[nt,nlayer,-2]
      v[nt,nlayer,-1] = v[nt,nlayer,-2]
      u[nt,nlayer,0] = 0
      v[nt,nlayer,0] = 0
      u[nt,nlayer,1] = u[nt,nlayer,0]
      v[nt,nlayer,1] = v[nt,nlayer,0]

   #---> h: Backward (corrector) in time using new value of u 
   for nlayer in range(nlayer_len):
      for nx in range(1,nx_len-1):
         h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * h[nt-1,nlayer,nx] * c[nx]*(u[nt,nlayer,nx+1]- u[nt,nlayer,nx]) )   
         h[nt,nlayer,nx] = h[nt,nlayer,nx] - u[nt,nlayer,nx]*(h[nt-1,nlayer,nx+1] - h[nt-1,nlayer,nx])*cm[nx]*dt
         h[nt,nlayer,nx] = h[nt,nlayer,nx] - h[nt-1,nlayer,nx]*u[nt,nlayer,nx]/rm[nx]
         h[nt,nlayer,nx] = h[nt,nlayer,nx] + (dt * M[nt,nlayer,nx])   		                              
      #--->Boundary conditions (ensure constant h at r=0 and r=+infty)
      h[nt,nlayer,0] = h[nt,nlayer,1]
      h[nt,nlayer,-1] = h[nt,nlayer,-2]

   #---> Values of interest (timeseries)
   PSO[nt] = g * ((ro1 * h[nt,0,0]) + (ro2 * h[nt,1,0]))   # central surface pressure
   vmax[nt] = npy.amax(v[nt,0,:]) 			   # initial maximum velocity

   SUM1 = 0;
   SUM2 = 0;
   for nx in range (result[0], result[-1]) :
      SUM1 = SUM1 + (h[nt,0,nx]*h[nt,0,nx] + eps*h[nt,1,nx]*h[nt,1,nx] + 2*eps*h[nt,0,nx]*h[nt,1,nx])*rm[nx]*math.pow(cm[nx],-1)
      SUM2 = SUM2 + (h[nt,0,nx]*(u[nt,0,nx]*u[nt,0,nx]+v[nt,0,nx]*v[nt,0,nx]) + eps*h[nt,1,nx]*(u[nt,1,nx]*u[nt,1,nx]+v[nt,1,nx]*v[nt,1,nx]))*rm[nx]*math.pow(cm[nx],-1)
   PE[nt] = math.pi*g*ro1*SUM1
   KE[nt] = math.pi*g*ro1*SUM2

################################################ OUTPUT (6) 
   if (nt == math.trunc(nt / ntplot) * ntplot):
    
      #PLOT h	
      plt.figure(figsize=(8,6))
      A = max([npy.amax(h[nt,0,:]),npy.amax(h[nt,1,:])])
      B = min([npy.amin(h[nt,0,:]),npy.amin(h[nt,1,:])])
      C = (A-B)/10
      plt.axis([0,r[-1]/1000,B-C ,A+C]) 
      plt.plot(r/1000, h[nt,0,:],linewidth=3.0, color='red')
      plt.plot(r/1000, h[nt,1,:],linewidth=3.0, color='blue')
      plt.xlabel('radius [km]',fontsize=14)
      plt.ylabel('h [m]',fontsize=14)
      plt.text(8000, A - C,"t="+str((nt)*dt/3600)+" hours")
      plt.text(8000, A - 2*C,"red: lower layer",color='red')
      plt.text(8000, A - 3*C,"blue: upper layer",color='blue')
      plt.savefig(DirecOutputh+"GradWindAdjustment-Run"+str(Runnumber)+"-h"+str(nt)+".png")
      plt.close()
  
      #PLOT v
      plt.figure(figsize=(8,6))
      A = max([npy.amax(v[nt,0,:]),npy.amax(v[nt,1,:])])
      B = min([npy.amin(v[nt,0,:]),npy.amin(v[nt,1,:])])
      C = (A-B)/10
      plt.axis([0,rm[-1]/1000,B-C ,A+C])   
      plt.plot(rm/1000, v[nt,0,:],linewidth=3.0, color='red')
      plt.plot(rm/1000, v[nt,1,:],linewidth=3.0, color='blue')
      plt.xlabel('radius [km]',fontsize=14)
      plt.ylabel('v [m/s]',fontsize=14)   
      plt.text(8000, A - C,"t="+str((nt)*dt/3600)+" hours") 
      plt.text(8000, A - 2*C,"red: lower layer",color='red')
      plt.text(8000, A - 3*C,"blue: upper layer",color='blue')
      plt.savefig(DirecOutputv+"GradWindAdjustment-Run"+str(Runnumber)+"-v"+str(nt)+".png")
      plt.close()

      #PLOT u
      plt.figure(figsize=(8,6))
      A = max([npy.amax(u[nt,0,:]),npy.amax(u[nt,1,:])])
      B = min([npy.amin(u[nt,0,:]),npy.amin(u[nt,1,:])])
      C = (A-B)/10
      plt.axis([0,rm[-1]/1000,B-C ,A+C])   
      plt.plot(rm/1000, u[nt,0,:],linewidth=3.0, color='red')
      plt.plot(rm/1000, u[nt,1,:],linewidth=3.0, color='blue')
      plt.xlabel('radius [km]',fontsize=14)
      plt.ylabel('u [m/s]',fontsize=14)   
      plt.text(8000,A - C,"t="+str((nt)*dt/3600)+" hours") 
      plt.text(8000,A - 2*C,"red: lower layer",color='red')
      plt.text(8000,A - 3*C,"blue: upper layer",color='blue')
      plt.savefig(DirecOutputu+"GradWindAdjustment-Run"+str(Runnumber)+"-v"+str(nt)+".png")
      plt.close()

#---> END TIME LOOP

########################################

# Compute local internal Rossby radius
RR = math.pow((1-eps)*g*Href,0.5)/(f_cor + (2 * v1max_i / RMW)) 

# Compute the gradient wind, vgrad at the end of the run and plot both vgrad and v
nt = len(time) - 1
for nx in range(1,nx_len-1):
   c1 = f_cor * rm[nx]
   gradphi0 = cm[nx] * ((geopot[nt,0,nx]-geopot[nt,0,nx-1]))   # pressur gradient calculated on rm-grid
   gradphi1 = cm[nx] * ((geopot[nt,1,nx]-geopot[nt,1,nx-1]))
   c2 = ((c1 * c1)+ (4*rm[nx]*gradphi0))
   if c2>0: vgrad[0,nx] = - 0.5 * (c1 - math.pow(c2,0.5))
   if c2<0: print('Problem with the gradient wind balance Layer 1 - Final step')
   c2 = ((c1 * c1)+ (4*rm[nx]*gradphi1))
   if c2>0: vgrad[1,nx] = - 0.5 * (c1 - math.pow(c2,0.5))
   if c2<0: print('Problem with the gradient wind balance Layer 2 - Final step')

# PLOT v and vgrad
plt.figure(figsize=(8,6))

plt.plot(rm[:]/1000, v[nt,0,:],linewidth=3.0, color='red')
plt.plot(rm[:]/1000, v[nt,1,:],linewidth=3.0, color='blue')
plt.plot(rm[:]/1000, vgrad[0,:],linewidth=1.0, color='red')
plt.plot(rm[:]/1000, vgrad[1,:],linewidth=1.0, color='blue')
plt.text(100,0.9*max(v[-1,0,:]), "t="+str(nt*dt/3600 +1)+" hours") 
plt.text(100,0.8*max(v[-1,0,:]),"red: lower layer (v:thick; vgrad: thin)",color='red')
plt.text(100,0.7*max(v[-1,0,:]),"blue: upper layer (v:thick; vgrad: thin)",color='blue')

plt.xlabel('radius [km]',fontsize=14)
plt.ylabel('v [m/s]',fontsize=14) 
 
plt.savefig(DirecOutputend+"GradientWind-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

#------------------------------

for nt in range (0,len(time)):
   PSO[nt] = (PSO[nt]-PSOref)/100.0
   PE[nt] = PE[nt] - math.pi*g*ro1*PE_ref
   KE[nt] = KE[nt] - math.pi*g*ro1*KE_ref

PE_max= npy.amax([npy.amax(PE[1:-1]),-npy.amin(PE[1:-1])])
KE_max= npy.amax([npy.amax(KE[1:-1]),-npy.amin(KE[1:-1])])
for nt in range (0,len(time)):
   PE[nt] = PE[nt]/PE_max
   KE[nt] = KE[nt]/KE_max
   
# PLOT time-evolution of central surface pressure (surface = lower boundary)
plt.figure(figsize=(8,6))
plt.plot(time*(1.0/3600.0), PSO,linewidth=2.0, color='black')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('surface pressure deficit at r=0 [hPa]',fontsize=14) 
plt.savefig(DirecOutputend+"PSO-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of max velocity
plt.figure(figsize=(8,6))
plt.plot(time*(1.0/3600.0), vmax,linewidth=2.0, color='black')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('maximum velocity in the system [m/s]',fontsize=14) 
plt.savefig(DirecOutputend+"MaxV-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of KE -PE
plt.figure(figsize=(8,6))
plt.plot(time[1:-1]*(1.0/3600.0), PE[1:-1],linewidth=3.0, color='blue')
plt.plot(time[1:-1]*(1.0/3600.0), KE[1:-1],linewidth=3.0, color='red')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('Energy density [J/m^3]',fontsize=14) 
plt.savefig(DirecOutputend+"TOT-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

#----------------------------------------------------------------------------------------------------------
