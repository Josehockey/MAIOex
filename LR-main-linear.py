import numpy as npy
import math as math
import matplotlib.pyplot as plt

# Solve linear shallow water equations in radial coordinate on a stretched grid, assuming axisymmetry
# Linear analysis over the initial background state imposed

DirecOutputVMAX = "/home/giovanni/SOAC/main3/Lin/VMAX/"
DirecOutputPMAX = "/home/giovanni/SOAC/main3/Lin/PMAX/"
DirecOutputWAVE = "/home/giovanni/SOAC/main3/Lin/WAVE/"

############################### INPUT PARAMETERS

#-----> Arts
Runnumber = 1   # number of the Run                                              
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
T_forcing = [2*3600.0, 4*3600.0, 6*3600.0, 8*3600.0, 10*3600.0]          #time scale forcing [s]
a = 500000.0                   # horizontal scale of the forcing [m]
Qmax = 0.4*Href                # amplitude of forcing ("Diabatic mass flux")               
r0 = 500000.0                  # radius of maximum forcing [m]

#-----> Initial
RMW = 500000.0     # radius of maximum initial state (velocity)
v1max_i = 20.0      # max. tang. wind lower layer                                   
v2max_i = 10.0      # max. tang. wind upper layer

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
PSO = npy.zeros((5,len(time)))                   #Timeseries of pressure in the centre
vmax = npy.zeros((5,len(time)))                  #Timeseries of maximum velocity 
PE = npy.zeros((5,len(time)))                    #Timeseries of the kinetic energy
KE = npy.zeros((5,len(time)))                    #Timeseries of potential energy
PE_ref = npy.zeros(5)			
KE_ref = npy.zeros(5)	

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
   for nt in range(len(time)):
      for nx in range(nx_len):
         if time[nt]<=T_forcing[i]: 
            M[nt,0,nx] = -(Qmax/T_forcing[i]) * math.exp(-math.pow((r[nx]-r0)/a,2))     	
            M[nt,1,nx] = (Qmax/(eps*T_forcing[i])) * math.exp(-math.pow((r[nx]-r0)/a,2)) 
         if time[nt]>T_forcing[i]: 
            M[nt,0,nx] = 0.0
            M[nt,1,nx] = 0.0

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
  
      #---> Compute the new height (LINEAR VERSION FULL EQUATIONS -> FORWARD H)
      for nlayer in range(nlayer_len):
         for nx in range(1,nx_len-1):
            h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * h[0,nlayer,nx] * c[nx]*(u[nt-1,nlayer,nx+1]- u[nt-1,nlayer,nx]) )   
            h[nt,nlayer,nx] = h[nt,nlayer,nx] - u[nt-1,nlayer,nx]*(h[0,nlayer,nx+1] - h[0,nlayer,nx])*cm[nx]*dt
            h[nt,nlayer,nx] = h[nt,nlayer,nx] - h[0,nlayer,nx]*u[nt-1,nlayer,nx]/rm[nx]
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
            v[nt,nlayer,nx] = v[nt-1,nlayer,nx] - ( dt * (f_cor + (v[0,nlayer,nx]/r[nx]) + (v[0,nlayer,nx+1] - v[0,nlayer,nx])*c[nx]) * u[nt-1,nlayer,nx])
            u[nt,nlayer,nx] = u[nt-1,nlayer,nx] - (dt * cm[nx] * (geopot[nt,nlayer,nx]-geopot[nt,nlayer,nx-1]))  
            u[nt,nlayer,nx] = u[nt,nlayer,nx] + (dt * (f_cor + (2*v[0,nlayer,nx]/r[nx])) * v[nt,nlayer,nx])
            u[nt,nlayer,nx] = u[nt,nlayer,nx] + (dt * ( ((v[0,nlayer,nx] * v[0,nlayer,nx])/rm[nx]) + (v[0,nlayer,nx]*f_cor) ) )
            v[nt,nlayer,nx] = v[nt-1,nlayer,nx] - ( dt * (f_cor + (v[0,nlayer,nx]/r[nx]) + (v[0,nlayer,nx+1] - v[0,nlayer,nx])*c[nx]) * u[nt,nlayer,nx])

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
            h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * h[0,nlayer,nx] * c[nx]*(u[nt,nlayer,nx+1]- u[nt,nlayer,nx]) )   
            h[nt,nlayer,nx] = h[nt,nlayer,nx] - u[nt,nlayer,nx]*(h[0,nlayer,nx+1] - h[0,nlayer,nx])*cm[nx]*dt
            h[nt,nlayer,nx] = h[nt,nlayer,nx] - h[0,nlayer,nx]*u[nt,nlayer,nx]/rm[nx]
            h[nt,nlayer,nx] = h[nt,nlayer,nx] + (dt * M[nt,nlayer,nx])         
         # Boundary conditions
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

for i in range(0,5):
   for nt in range (0,len(time)):
      PSO[i,nt] = (PSO[i,nt]-PSOref)/100.0
      PE[i,nt] = PE[i,nt] - math.pi*g*ro1*PE_ref[i]
      KE[i,nt] = KE[i,nt] - math.pi*g*ro1*KE_ref[i]

'''
PE_max = npy.zeros(5)
KE_max = npy.zeros(5)
for i in range(0,5):
   PE_max[i]= npy.amax([npy.amax(PE[i,1:-1]),-npy.amin(PE[i,1:-1])])
   KE_max[i]= npy.amax([npy.amax(KE[i,1:-1]),-npy.amin(KE[i,1:-1])])
   for nt in range (0,len(time)):
      PE[i,nt] = PE[i,nt]/PE_max[i]
      KE[i,nt] = KE[i,nt]/KE_max[i]
'''

# PLOT time-evolution of central surface pressure (surface = lower boundary)
plt.figure(figsize=(8,6))
for t in range (0,5): plt.plot(time*(1.0/3600.0), PSO[t,:],linewidth=2.0)
plt.legend(['Sp 300km','Sp 400km','Sp 500km','Sp 700km','Sp 800km'], loc='upper right')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('surface pressure deficit at r=0 [hPa]',fontsize=14) 
plt.savefig(DirecOutputPMAX+"ps0-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of max velocity
plt.figure(figsize=(8,6))
for t in range (0,5): plt.plot(time*(1.0/3600.0), vmax[t,:],linewidth=2.0)
plt.legend(['Sp 300km','Sp 400km','Sp 500km','Sp 700km','Sp 800km'], loc='upper right')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('maximum velocity in the system [m/s]',fontsize=14) 
plt.savefig(DirecOutputVMAX+"MaxV-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of PE
plt.figure(figsize=(8,6))
for t in range (0,5): plt.plot(time[1:-1]*(1.0/3600.0), PE[t,1:-1],linewidth=2.0)
plt.legend(['Sp 300km','Sp 400km','Sp 500km','Sp 700km','Sp 800km'], loc='upper right')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('Potential Energy density normalize []',fontsize=14) 
plt.savefig(DirecOutputWAVE+"PE-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of KE
plt.figure(figsize=(8,6))
for t in range (0,5): plt.plot(time[1:-1]*(1.0/3600.0), KE[t,1:-1],linewidth=2.0)
plt.legend(['Sp 300km','Sp 400km','Sp 500km','Sp 700km','Sp 800km'], loc='upper right')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('Kinetic Energy density normalized []',fontsize=14) 
plt.savefig(DirecOutputWAVE+"KE-Run"+str(Runnumber)+".png")
plt.show()
plt.close()
#--------------------------------------------------------- FROM NOW ON STUDY BASED ON THE MODEL ------------------------------------------------------------
