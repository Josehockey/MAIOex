############################### OUTPUT PATH

import numpy as npy
import math as math
import matplotlib.pyplot as plt

# Solve shallow water equations in radial coordinate on a stretched grid, assuming axisymmetry
# Non linear analysis over the initial background state imposed

DirecOutputVMAX = "/home/giovanni/SOAC/main3/NonLin/VMAX/"
DirecOutputPMAX = "/home/giovanni/SOAC/main3/NonLin/PMAX/"
DirecOutputWAVE = "/home/giovanni/SOAC/main3/NonLin/WAVE/"

############################### INPUT PARAMETERS

#-----> Arts
Runnumber = 10   # number of the Run                                              
ntplot = 1800      # frequency of output in steps (-> Look at dt for conversion)

#-----> Pysical
g = 9.81
f_cor = 0.0001     # 45 degree
eps = 0.9      
ro1 = 1.0
ro2 = eps * ro1
Href = 5000.0
nlayer_len = 2
PSOref =  g*((ro1*Href)+(ro2*Href))        # central surface pressure : reference state    

#-----> Forcing
T_forcing = 6*3600.0          #time scale forcing [s]
a = 500000.0                 # horizontal scale of the forcing [m]
Qmax = 0.5*Href               # amplitude of forcing ("Diabatic mass flux")               
r0 = [100000.0, 300000.0, 500000.0, 700000.0, 900000.0]              # radius of maximum forcing [m]

#-----> Initial
RMW = 500000.0     # radius of maximum initial state (velocity)
v1max_i = 0.0      # max. tang. wind lower layer                                   
v2max_i = 0.0      # max. tang. wind upper layer

############################### GRIDS

#------> Time
dt = 2            # time step in seconds
end = 24*3600     # 24 total time of integration                                  
time = npy.arange(0, end, dt)           

#-----> Space
nx_len = 125      # number of points
dx = 1            # grid risolution
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
PSO = npy.zeros((5,len(time)))                   # Timeseries of pressure in the centre
vmax = npy.zeros((5,len(time)))                  # Timeseries of maximum velocity 
PE = npy.zeros((5,len(time)))                    # Timeseries of the potential energy
KE = npy.zeros((5,len(time)))                    # Timeseries of kinetic energy
PE_ref = npy.zeros(5)			         # Potential energy at t=0
KE_ref = npy.zeros(5)				 # Kinetic energy at t=0 		

###################### INITIAL CONDITIONS

for i in range(0,5) :

   #-----> Rankine vortex
   for nx in range(0, nx_len):
      if r[nx] <= RMW: 
         v[0,0,nx] = v1max_i * r[nx] / RMW
         v[0,1,nx] = v2max_i * r[nx] / RMW
  
      if rm[nx] > RMW: 
         v[0,0,nx] = v1max_i * RMW / r[nx]
         v[0,1,nx] = v2max_i * RMW / r[nx]
  
   for nlayer in range(nlayer_len):
      for nx in range(nx_len):
         u[0,nlayer,nx] = 0                        # Initial radial velocity set to zero.

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
   Modified1 = npy.amin(npy.where(r > 1500000))
   Modified2 = npy.amin(npy.where(r > 2000000))
   
   for nt in range(len(time)):
      for nx in range(Modified2):
         if nx < Modified1:
            if time[nt]<=T_forcing: 
               M[nt,0,nx] = -(Qmax/T_forcing) * math.exp(-math.pow((r[nx]-r0[i])/a,2))     	
               M[nt,1,nx] = (Qmax/(eps*T_forcing)) * math.exp(-math.pow((r[nx]-r0[i])/a,2)) 
            if time[nt]>T_forcing: 
               M[nt,0,nx] = 0.0
               M[nt,1,nx] = 0.0
         else:
            M[nt,0,nx] = (1-math.exp(-100*(nx-Modified2+1)))*M[nt,0,Modified1]

   plt.figure(figsize=(8,6))
   plt.plot(M[3,0,:], r)
   plt.plot(M[3,1,:], r)
   plt.show()

   #-----> Variables of interest: initialization
   PSO[i,0] = g * ((ro1 * h[0,0,0]) + (ro2 * h[0,1,0]))    # central surface pressure : initial state
   vmax[i,0] = npy.amax(v[0,0,:]) 			   # initial maximum velocity
       
   #-----> Energy inspection 
   start = 5000000     #starting inspection [m]
   end = 10000000      #end inspection [m]
   result1 = npy.where(r < end)
   result2 = npy.where(r > start)
   result = npy.arange(npy.amin(result2), npy.amax(result1), 1)
   for nx in range (result[0], result[-1]):
      PE_ref[i] = PE_ref[i] + (h[0,0,nx]*h[0,0,nx] + eps*h[0,1,nx]*h[0,1,nx] + 2*eps*h[0,0,nx]*h[0,1,nx])*rm[nx]*math.pow(cm[nx],-1)
      KE_ref[i] = KE_ref[i] + (h[0,0,nx]*(u[0,0,nx]*u[0,0,nx]+v[0,0,nx]*v[0,0,nx]) + eps*h[0,1,nx]*(u[0,1,nx]*u[0,1,nx]+v[0,1,nx]*v[0,1,nx]))*rm[nx]*math.pow(cm[nx],-1)
  
##################################### ALGORITHM 

   for nt in range(1,len(time)):
  
      #---> Compute the new height (NON LINEAR VERSION FULL EQUATIONS -> FORWARD H)
      for nlayer in range(nlayer_len):
         for nx in range(1,nx_len-1):
            h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * h[nt-1,nlayer,nx] * c[nx]*(u[nt-1,nlayer,nx+1]- u[nt-1,nlayer,nx]) )   
            h[nt,nlayer,nx] = h[nt,nlayer,nx] - u[nt-1,nlayer,nx]*(h[nt-1,nlayer,nx+1] - h[nt-1,nlayer,nx])*cm[nx]*dt
            h[nt,nlayer,nx] = h[nt,nlayer,nx] - h[nt-1,nlayer,nx]*u[nt-1,nlayer,nx]/rm[nx]
            h[nt,nlayer,nx] = h[nt,nlayer,nx] + (dt * M[nt,nlayer,nx])   		                              
         #--->Boundary conditions (ensure constant h at r=0 and r=+infty)
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

      # Boundary conditions -> Continuty at the boundary and zero in the centre
      for nlayer in range(nlayer_len):
         u[nt,nlayer,-1] = u[nt,nlayer,-2]
         v[nt,nlayer,-1] = v[nt,nlayer,-2]
         u[nt,nlayer,0] = 0
         v[nt,nlayer,0] = 0
         u[nt,nlayer,1] = u[nt,nlayer,0]
         v[nt,nlayer,1] = v[nt,nlayer,0]

      # Backward (corrector) time step in h
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
      PSO[i,nt] = g * ((ro1 * h[nt,0,0]) + (ro2 * h[nt,1,0]))   # central surface pressure
      vmax[i,nt] = npy.amax(v[nt,0,:]) 			   # initial maximum velocity
      SUM1 = 0;
      SUM2 = 0;
      for nx in range (result[0], result[-1]) :
         SUM1 = SUM1 + (h[nt,0,nx]*h[nt,0,nx] + eps*h[nt,1,nx]*h[nt,1,nx] + 2*eps*h[nt,0,nx]*h[nt,1,nx])*rm[nx]*math.pow(cm[nx],-1)
         SUM2 = SUM2 + (h[nt,0,nx]*(u[nt,0,nx]*u[nt,0,nx]+v[nt,0,nx]*v[nt,0,nx])+eps*h[nt,1,nx]*(u[nt,1,nx]*u[nt,1,nx]+v[nt,1,nx]*v[nt,1,nx]))*rm[nx]*math.pow(cm[nx],-1)
      PE[i,nt] = math.pi*g*ro1*SUM1
      KE[i,nt] = math.pi*g*ro1*SUM2

   print((i+1)/5.0,'%')

#---> END TIME LOOP
#########################################################################################
Legend = ['ro = 100km','ro = 300km','ro = 500km','ro = 700km','ro = 900km']
#['Qmax = 500m','Qmax = 1500m','Qmax = 2500m','Qmax = 3500m','Qmax = 4500m']
#['ro = 100km','ro = 300km','ro = 500km','ro = 700km','ro = 900km']
#['a = 100km','a = 300km','a = 500km','a = 700km','a = 900km']

for i in range(0,5):
   for nt in range (0,len(time)):
      PSO[i,nt] = (PSO[i,nt]-PSOref)/100.0
      PE[i,nt] = PE[i,nt] - math.pi*g*ro1*PE_ref[i]
      KE[i,nt] = KE[i,nt] - math.pi*g*ro1*KE_ref[i]

PEresult = npy.zeros(5)
KEresult = npy.zeros(5)
for i in range(0,5): 
   flag1 = 0
   flag2 = 0
   for nt in range(1,len(time)):
      if abs(PE[i,nt]) > 0.001*npy.amax([npy.amax(PE[i,1:-1]), -npy.amin(PE[i,1:-1])]) and flag1==0 :
         flag1 = 1 
         print(PE[i,0:20])
         PEresult[i] = round((time[nt]*(1.0/3600.0)),2)
      if abs(KE[i,nt]) > 0.001*npy.amax([npy.amax(KE[i,1:-1]), -npy.amin(KE[i,1:-1])]) and flag2==0 :
         flag2 = 1 
         print(KE[i,0:20])
         KEresult[i] = round((time[nt]*(1.0/3600.0)),2)

# PLOT time-evolution of central surface pressure (surface = lower boundary)
plt.figure(figsize=(8,6))
for t in range (0,5): plt.plot(time*(1.0/3600.0), PSO[t,:],linewidth=2.0)
plt.legend(Legend, loc='upper right')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('surface pressure deficit at r=0 [hPa]',fontsize=14) 
plt.savefig(DirecOutputPMAX+"ps0-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of max velocity
plt.figure(figsize=(8,6))
for t in range (0,5): plt.plot(time*(1.0/3600.0), vmax[t,:],linewidth=2.0)
plt.legend(Legend, loc='upper right')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('maximum velocity in the system [m/s]',fontsize=14) 
plt.savefig(DirecOutputVMAX+"MaxV-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of PE
plt.figure(figsize=(8,6))
for t in range (0,5): plt.plot(time[1:-1]*(1.0/3600.0), PE[t,1:-1],linewidth=2.0)
lab1 = plt.legend(Legend, loc='upper right')
lab2 = plt.legend(PEresult, loc='upper left')
plt.gca().add_artist(lab1)
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('Potential Energy fluctuations  [J]',fontsize=14) 
plt.savefig(DirecOutputWAVE+"PE-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of KE
plt.figure(figsize=(8,6))
for t in range (0,5): plt.plot(time[1:-1]*(1.0/3600.0), KE[t,1:-1],linewidth=2.0)
lab1 = plt.legend(Legend, loc='upper right')
lab2 = plt.legend(KEresult, loc='upper left')
plt.gca().add_artist(lab1)
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('Kinetic Energy fluctuations  [J]',fontsize=14) 
plt.savefig(DirecOutputWAVE+"KE-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

#--------------------------------------------------------- FROM NOW ON STUDY BASED ON THE MODEL ------------------------------------------------------------
