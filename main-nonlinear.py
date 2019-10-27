import numpy as npy
import math as math
import matplotlib.pyplot as plt

# Solve linear shallow water equations in radial coordinate on a stretched grid, assuming axisymmetry
# Non linear analysis over the initial background state imposed

DirecOutputh = "/home/giovanni/SOAC/main2/Nlin/h/"
DirecOutputv = "/home/giovanni/SOAC/main2/Nlin/v/"
DirecOutputu = "/home/giovanni/SOAC/main2/Nlin/u/"
DirecOutputend = "/home/giovanni/SOAC/main2/Nlin/END/"

###############################  INPUT PARAMETERS

#-----> Arts
Runnumber = 1010   # number of the Run                                              
ntplot = 1800      # frequency of output in steps (-> Look at dt for conversion)

#-----> Pysical
g = 9.81
f_cor = 0.0001   #45 degree
eps = 0.9      
ro1 = 1.0
ro2 = eps * ro1
Href = 5000.0
nlayer_len = 2

#-----> Forcing
T_forcing= 6 * 3600.0          #time scale forcing [s]
a = 500000.0                   # horizontal scale of the forcing [m]
Qmax = Href/(24*3600)    # amplitude of forcing ("Diabatic mass flux") normalize with time of integration               
r0 = 500000.0                  # radius of maximum forcing [m]

#-----> Initial
RMW = 500000.0     # radius of maximum initial state (velocity)
v1max_i = 20.0      # max. tang. wind lower layer                                   
v2max_i = 10.0      # max. tang. wind upper layer

############################### GRIDS

#------> Time
dt = 1  # time step in seconds
end = 24*3600 # 24 total time of integration                                  
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

#-----> Velocities
u = npy.zeros((len(time),nlayer_len,nx_len))
v = npy.zeros((len(time),nlayer_len,nx_len))
h = npy.zeros((len(time),nlayer_len,nx_len))
M = npy.zeros((len(time),nlayer_len,nx_len))
geopot = npy.zeros((len(time),nlayer_len,nx_len))                               

vgrad = npy.zeros((nlayer_len,nx_len))       #--> Time indipendent 

ps0 = npy.zeros(len(time))                   #--> Timeseries of pressure in the centre
vmax = npy.zeros(len(time))   
rmax = npy.zeros(len(time)) 

###################### INITIAL CONDITIONS

#-----> Rankine vortex
for nx in range(0, nx_len):
   if rm[nx] <= RMW: 
      v[0,0,nx] = v1max_i * rm[nx] / RMW
      v[0,1,nx] = v2max_i * rm[nx] / RMW
  
   if rm[nx] > RMW: 
      v[0,0,nx] = v1max_i * RMW / rm[nx]
      v[0,1,nx] = v2max_i * RMW / rm[nx]
  
for nlayer in range(nlayer_len):
   for nx in range(nx_len):
      u[0,nlayer,nx] = 0							    #---> Initial radial velocity set to zero.

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
         M[nt,0,nx] = -(Qmax) * math.exp(-math.pow((r[nx]-r0)/a,2))     	
         M[nt,1,nx] = (Qmax/(eps)) * math.exp(-math.pow((r[nx]-r0)/a,2)) 
      if time[nt]>T_forcing: 
         M[nt,0,nx] = 0.0
         M[nt,1,nx] = 0.0

#Values of interest -> Initial conditions

ps0ref =  g * ((ro1 * Href) + (ro2 * Href))        # central surface pressure : reference state         
ps0[0] = g * ((ro1 * h[0,0,0]) + (ro2 * h[0,1,0])) # central surface pressure : initial state

vmax[0] = npy.amax(v[0,0,:]) 			   # initial maximum velocity
index = npy.where(v[0,0,:] == vmax[0])  
Un = npy.zeros([1,1])
if npy.shape(index) == npy.shape(Un) :		   # initial ray of maximum velocity
   rmax[0] = r[index]
else :
   rmax[0] = 0

##################################### ALGORITHM 

for nt in range(1,len(time)):
 
   time[nt] = time[nt] / 3600.0  # time in hours
  
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

   # v: forward (predictor) in time using previous value of u
   # u: Backward in time using new values of h and v
   # v: Backward (corrector) in time using new value of u

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

#Values of interest -> Timeseries
   ps0[nt] = g * ((ro1 * h[nt,0,0]) + (ro2 * h[nt,1,0])) # central surface pressure
   vmax[nt] = npy.amax(v[nt,0,:]) 			   # initial maximum velocity
   index = npy.where(v[nt,0,:] == vmax[nt])  
   if npy.shape(index) == npy.shape(Un):		   # initial ray of maximum velocity
      rmax[nt] = r[index]
   else:
      rmax[nt] = 0
  
#------------> BEGIN OUTPUT 
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
      plt.text(100, A - C,"t="+str((nt)*dt/3600)+" hours")
      plt.text(100, A - 2*C,"red: lower layer",color='red')
      plt.text(100, A - 3*C,"blue: upper layer",color='blue')
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
      plt.text(100, A - C,"t="+str((nt)*dt/3600)+" hours") 
      plt.text(100, A - 2*C,"red: lower layer",color='red')
      plt.text(100, A - 3*C,"blue: upper layer",color='blue')
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
      plt.text(100,A - C,"t="+str((nt)*dt/3600)+" hours") 
      plt.text(100,A - 2*C,"red: lower layer",color='red')
      plt.text(100,A - 3*C,"blue: upper layer",color='blue')
      plt.savefig(DirecOutputu+"GradWindAdjustment-Run"+str(Runnumber)+"-v"+str(nt)+".png")
      plt.close()

#---> END OUTPUT 
#---> END TIME LOOP

#---> END OUTPUT 
#---> END TIME LOOP
#########################################################################################


# compute local internal Rossby radius
RR = math.pow((1-eps)*g*Href,0.5)/(f_cor + (2 * v1max_i / RMW)) 

nt = len(time) - 1 
print ("number  Dps(r=0)     a  Rossby radius  v1max_i  v2max_i   RMW      Qmax/Href   T_forcing   Time")
print ("%5.0f, %8.2f, %8.2f, %8.2f,%8.2f,%8.2f, %8.2f, %8.2f, %8.2f, %8.2f")% (Runnumber,ps0[nt],a/1000,RR/1000,v1max_i,v2max_i,RMW/1000,Qmax/Href,T_forcing/3600.,nt*dt/3600.)

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
 
plt.savefig(DirecOutputend+"GradientWind-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

T = npy.arange(0, end, dt) 
for i in range (len(ps0)):
   ps0[i] = (ps0[i]-ps0ref)/100.0
   rmax[i] = rmax[i]/1000.0

# PLOT time-evolution of central surface pressure (surface = lower boundary)
plt.figure(figsize=(8,6))
plt.plot(T, ps0,linewidth=2.0, color='black')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('surface pressure deficit at r=0 [hPa]',fontsize=14) 
plt.savefig(DirecOutputend+"ps0-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of max velocity
plt.figure(figsize=(8,6))
plt.plot(T, vmax,linewidth=2.0, color='black')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('maximum velocity in the system [m/s]',fontsize=14) 
plt.savefig(DirecOutputend+"MaxV-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

# PLOT time-evolution of rmax velocity
plt.figure(figsize=(8,6))
plt.plot(T, rmax,linewidth=2.0, color='black')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('ray of maximum velocity [km]',fontsize=14) 
plt.savefig(DirecOutputend+"MaxR-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

#--------------------------------------------------------- FROM NOW ON STUDY BASED ON THE MODEL ------------------------------------------------------------
