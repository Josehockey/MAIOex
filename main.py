import numpy as npy
import math as math
import matplotlib.pyplot as plt

# Solve linear shallow water equations in radial coordinate on a stretched grid, assuming axisymmetry

DirecOutputh = "/home/giovanni/SOAC/main/h/"
DirecOutputv = "/home/giovanni/SOAC/main/v/"
DirecOutputend = "/home/giovanni/SOAC/main/END/"
DirecOutputu = "/home/giovanni/SOAC/main/u/"


def pe(h,cm):
   PE = 0.5*g*ro1*(h[0]*h[0])/cm + 0.5*g*ro2*(h[1]*h[1]+2*h[1]*h[0])/cm
   return PE
def ke(u,v,h,cm):
   KE =0.5*h[0]*(ro1*(v[0]**2+u[0]**2)/cm)+0.5*h[1]*(ro1*(v[1]**2+u[1]**2)/cm)
   return KE
###############################  INPUT PARAMETERS

#-----> Arts
Runnumber = 1 # number of the Run                                              
ntplot = 1800 # frequency of output in steps (-> Look at dt for conversion)

#-----> Pysical
g = 9.81
f_cor = 0.0001   
eps = 0.9       # =ro2/ro1
ro1 = 1.0
ro2 = eps * ro1
Href = 5000.0   # reference depth fluid [m]
nlayer_len = 2

#-----> Forcing
T_forcing= 6 * 3600.0 #time scale forcing [s]
a = 500000.0  # horizontal scale of the forcing [m]
Qmax = 0.2 * Href  # amplitude of forcing ("Diabatic mass flux")               #Unit do not metter, it's proportional to the volume.
r0 = 500000.0  # radius of maximum forcing [m]
Hpert = Qmax

#-----> Initial
RMW = 500000.0  # radius of maximum initial state (velocity)
v1max_i = 0.0  # max. tang. wind lower layer                                   
v2max_i = 0.0  # max. tang. wind upper layer

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
rv = npy.zeros((len(time),nlayer_len,nx_len))                                     ###WHAT'S THIS?  ---->(RELATIVE VORTICITY??)
zeta = npy.zeros((len(time),nlayer_len,nx_len))                                   ###AND THIS?     ---->(POTENTIAL VORTICITY??)
h = npy.zeros((len(time),nlayer_len,nx_len))
M = npy.zeros((len(time),nlayer_len,nx_len))
geopot = npy.zeros((len(time),nlayer_len,nx_len))

vgrad = npy.zeros((nlayer_len,nx_len))       #--> Time indipendent 
ps0 = npy.zeros(len(time))                   #--> Timeseries of pressure in the centre

###################### INITIAL CONDITIONS

#-----> Rankine vortex
for nx in range(0, nx_len):
   if rm[nx] <= RMW: 
      v[0,0,nx] = v1max_i * rm[nx] / RMW
      v[0,1,nx] = v2max_i * rm[nx] / RMW
      zeta[0,0,nx] = 2 * v1max_i / RMW
      zeta[0,1,nx] = 2 * v2max_i / RMW						#---> Why zeta is given by this?
  
   if rm[nx] > RMW: 
      v[0,0,nx] = v1max_i * RMW / rm[nx]
      v[0,1,nx] = v2max_i * RMW / rm[nx]
      zeta[0,0,nx] = 0.0
      zeta[0,1,nx] = 0.0

for nlayer in range(nlayer_len):
   for nx in range(1,nx_len-1):
      rv[0,nlayer,nx] = rm[nx] * v[0,nlayer,nx]                                     #---> Initial relative vorticity?
  
for nlayer in range(nlayer_len):
   for nx in range(nx_len):
      u[0,nlayer,nx] = 0							    #---> Initial radial velocity set to zero.

#-----> Gradient wind balance at t=0
h[0,0,-1] = Href 
h[0,1,-1] = Href
for nx in range(1,nx_len):
   dphi1 = (((v[0,0,-nx]*v[0,0,-nx])/rm[-nx]) + (f_cor*v[0,0,-nx])) * labda * (rm[-nx]+b)        #HERE ALSO MULTIPLY dx. BUT dx IS TAKEN AS 1. 
   dphi2 = (((v[0,1,-nx]*v[0,1,-nx])/rm[-nx]) + (f_cor*v[0,1,-nx])) * labda * (rm[-nx]+b) 
   dh2 = (1/((1-eps)*g)) * (dphi2 - dphi1)
   dh1 = (dphi1/g) - (eps*dh2)
   h[0,0,-nx-1] = h[0,0,-nx]-dh1
   h[0,1,-nx-1] = h[0,1,-nx]-dh2                                                    #---> So in the centre at beginnin H = Href 
                                                                                #[I WOULD LIKE TO DO Href OUTSIDE THE CYCLONE]
for nx in range (nx_len):
   geopot[0,0,nx] = g * (h[0,0,nx] + (eps * h[0,1,nx] ))
   geopot[0,1,nx] = g * (h[0,0,nx] + h[0,1,nx] )                                  #MISLUKE? h[0,1]+h[0,1]??

ps0ref =  g * ((ro1 * Href) + (ro2 * Href)) # central surface pressure : reference state            ###I don't get, you calculate ps0ref but it's ps0[0]
ps0[0] = g * ((ro1 * h[0,0,0]) + (ro2 * h[0,1,0])) # central surface pressure : initial state

#-----> Specify forcing
for nt in range(len(time)):
   for nx in range(nx_len):
      if time[nt]<=T_forcing: 
         M[nt,0,nx] = -(Qmax/T_forcing) * math.exp(-math.pow((r[nx]-r0)/a,2))     #*(time[nt]/T_forcing)        # layer 1			
         M[nt,1,nx] = (Qmax/(eps*T_forcing)) * math.exp(-math.pow((r[nx]-r0)/a,2))   #*(time[nt]/T_forcing)   # layer 1
      if time[nt]>T_forcing: 
         M[nt,0,nx] = 0.0
         M[nt,1,nx] = 0.0

##################################### ALGORITHM 

for nt in range(1,len(time)):
 
   time[nt] = time[nt] / 3600.0  # time in hours    (NOT NECESSARY)
  
   #---> Compute the new height
   for nlayer in range(nlayer_len):
      for nx in range(1,nx_len-1):
         h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * (Href / r[nx]) * ( c[nx]*((rm[nx+1]*u[nt-1,nlayer,nx+1])-(rm[nx]*u[nt-1,nlayer,nx]))) )   #This should be eq. 3 but 
         h[nt,nlayer,nx] = h[nt,nlayer,nx] + (dt * M[nt,nlayer,nx])     		                                #miss advection and H is taken constant as Href. 
      #--->Boundary conditions (ensure constant h at r=0 and r=+infty)
      h[nt,nlayer,0] = h[nt,nlayer,1]
      h[nt,nlayer,-1] = h[nt,nlayer,-2]

   #---> Compute the new geopotential
   for nx in range(nx_len):
      geopot[nt,0,nx] = g * (h[nt,0,nx] + (eps * h[nt,1,nx] ))
      geopot[nt,1,nx] = g * (h[nt,0,nx] + h[nt,1,nx] )
  
# Compute the gradient wind, vgrad at x=1,2,3...   , i.e at the same grid points as geopotential and height (MISLUKE)
   for nx in range(1,nx_len-1):
      c1 = f_cor * r[nx]
      gradphi0 = c[nx] * (geopot[nt,0,nx+1]-geopot[nt,0,nx-1])/2  # radial gradient geopotential calculated on rm-grid
      gradphi1 = c[nx] * (geopot[nt,1,nx+1]-geopot[nt,1,nx-1])/2
      c2 = ((c1 * c1)+ (4*r[nx]*gradphi0))
      if c2>=0: vgrad[0,nx] = - 0.5 * (c1 - math.pow(c2,0.5))
      if c2<0: print('Problem with the gradient wind balance Layer 1')
      c2 = ((c1 * c1)+ (4*r[nx]*gradphi1))
      if c2>=0: vgrad[1,nx] = - 0.5 * (c1 - math.pow(c2,0.5))
      if c2<0: print('Problem with the gradient wind balance Layer 2')
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
      u[nt,nlayer,-1] = u[nt,nlayer,-2]
      v[nt,nlayer,-1] = v[nt,nlayer,-2]
      u[nt,nlayer,0] = 0
      v[nt,nlayer,0] = 0

# Backward (corrector) time step in h
   for nlayer in range(nlayer_len):
      for nx in range(2,nx_len-1):
         h[nt,nlayer,nx] = h[nt-1,nlayer,nx] - (dt * (Href / r[nx]) * ( c[nx]*((rm[nx+1]*u[nt,nlayer,nx+1])-(rm[nx]*u[nt,nlayer,nx]))) )   # h on r-grid
         h[nt,nlayer,nx] = h[nt,nlayer,nx] + (dt * M[nt,nlayer,nx])     

# Boundary conditions
   for nlayer in range(nlayer_len):
      h[nt,nlayer,0] = h[nt,nlayer,1]
      h[nt,nlayer,-1] = h[nt,nlayer,-2]
#Energy
   for nx in range(1,nx_len-1):
      KE[nt]=KE[nt] + ke(u[nt,:,nx],v[nt,:,nx],h[nt,:,nx],cm[nx])
      PE[nt]=PE[nt] + pe(h[nt,:,nx],cm[nx])

   ps0[nt] = g * ((ro1 * h[nt,0,0]) + (ro2 * h[nt,1,0])) # central surface pressure
   #ps0[nt] = (ps0[nt]-ps0ref)

#------------> BEGIN OUTPUT 
   if (nt == math.trunc(nt / ntplot) * ntplot):
    
      #PLOT h	
      plt.figure(figsize=(8,6))
      A = max([npy.amax(h[nt,0,:]),npy.amax(h[nt,1,:])])
      B = min([npy.amin(h[nt,0,:]),npy.amin(h[nt,1,:])])
      plt.axis([0,r[-1]/1000,B-100 ,A+100]) 
      plt.plot(r/1000, h[nt,0,:],linewidth=3.0, color='red')
      plt.plot(r/1000, h[nt,1,:],linewidth=3.0, color='blue')
      plt.xlabel('radius [km]',fontsize=14)
      plt.ylabel('h [m]',fontsize=14)
      plt.text(100, A - 1*(A-B)/100,"t="+str((nt)*dt/3600)+" hours")
      plt.text(100, A - 4*(A-B)/100,"red: lower layer",color='red')
      plt.text(100, A - 7*(A-B)/100,"blue: upper layer",color='blue')
      plt.savefig(DirecOutputh+"GradWindAdjustment-Run"+str(Runnumber)+"-h"+str(nt)+".png")
      plt.close()
  
      #PLOT v
      plt.figure(figsize=(8,6))
      A = max([npy.amax(v[nt,0,:]),npy.amax(v[nt,1,:])])
      B = min([npy.amin(v[nt,0,:]),npy.amin(v[nt,1,:])])
      plt.axis([0,rm[-1]/1000,B-100 ,A+100])   
      plt.plot(rm/1000, v[nt,0,:],linewidth=3.0, color='red')
      plt.plot(rm/1000, v[nt,1,:],linewidth=3.0, color='blue')
      plt.xlabel('radius [km]',fontsize=14)
      plt.ylabel('v [m/s]',fontsize=14)   
      plt.text(100, A - 1*(A-B)/100,"t="+str((nt)*dt/3600)+" hours") 
      plt.text(100, A - 4*(A-B)/100,"red: lower layer",color='red')
      plt.text(100, A - 7*(A-B)/100,"blue: upper layer",color='blue')
      plt.savefig(DirecOutputv+"GradWindAdjustment-Run"+str(Runnumber)+"-v"+str(nt)+".png")
      plt.close()

      #PLOT u
      plt.figure(figsize=(8,6))
      A = max([npy.amax(v[nt,0,:]),npy.amax(v[nt,1,:])])
      B = min([npy.amin(v[nt,0,:]),npy.amin(v[nt,1,:])])
      plt.axis([0,rm[-1]/1000,B-100 ,A+100])   
      plt.plot(rm/1000, u[nt,0,:],linewidth=3.0, color='red')
      plt.plot(rm/1000, u[nt,1,:],linewidth=3.0, color='blue')
      plt.xlabel('radius [km]',fontsize=14)
      plt.ylabel('u [m/s]',fontsize=14)   
      plt.text(100,A - 1*(A-B)/100,"t="+str((nt)*dt/3600)+" hours") 
      plt.text(100,A - 4*(A-B)/100,"red: lower layer",color='red')
      plt.text(100,A - 7*(A-B)/100,"blue: upper layer",color='blue')
      plt.savefig(DirecOutputu+"GradWindAdjustment-Run"+str(Runnumber)+"-v"+str(nt)+".png")
      plt.close()

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
# PLOT time-evolution of central surface pressure (surface = lower boundary)
plt.figure(figsize=(8,6))
#plt. axis([0,len(time)*dt/3600,-5,5])
plt.plot(T, ps0,linewidth=2.0, color='black')
plt.xlabel('time [hours]',fontsize=14)
plt.ylabel('surface pressure deficit at r=0 [hPa]',fontsize=14) 

plt.savefig(DirecOutputend+"ps0-Run"+str(Runnumber)+".png")
plt.show()
plt.close()

#--------------------------------------------------------- FROM NOW ON STUDY BASED ON THE MODEL ------------------------------------------------------------

PE_F1 = 0
PE_F2 = 0  
PE_ref1 = 0
PE_ref2 = 0
for nx in range (0, nx_len):
   PE_F1 = PE_F1 + ro1*h[-1,0,nx]*g/c[nx]
   PE_F2 = PE_F2 + ro2*h[-1,1,nx]*g/c[nx]
   PE_ref1 = PE_ref1 + ro1*h[0,0,nx]*g/c[nx]
   PE_ref2 = PE_ref2 + ro2*h[0,1,nx]*g/c[nx]

KE_F1 = 0
KE_F2 = 0
KE_ref1 = 0 
KE_ref2 = 0 
for nx in range (0,nx_len):
   KE_F1 = KE_F1 + ro1*(v[-1,0,nx]*v[-1,0,nx] + u[-1,0,nx]*u[-1,0,nx])/cm[nx]
   KE_F2 = KE_F2 + ro2*(v[-1,1,nx]*v[-1,1,nx] + u[-1,1,nx]*u[-1,1,nx])/cm[nx]
   KE_ref1 = KE_ref1 + ro1*(v[0,0,nx]*v[0,0,nx] + u[0,0,nx]*u[0,0,nx])/cm[nx]
   KE_ref2 = KE_ref2 + ro2*(v[0,1,nx]*v[0,1,nx] + u[0,1,nx]*u[0,1,nx])/cm[nx]

print(PE_ref1-PE_F1, PE_ref2-PE_F2, KE_F1-KE_ref1, KE_F2-KE_ref2 )
