import matplotlib.pyplot as plt
import math
import numpy as np

def Fourier(S, T, F) :
  #[Signal]    #S is the data set
  #[Time]      #T is an array with start time,      finish time,      time intervall
  #[Frequency] #F is an array with start frequency, finish frequency, frequency intervall
  Time = np.arange(T[0], T[1], T[2]) ;
  Freq = np.arange(F[0], F[1], F[2]) ;
  FT = [0]*len(Freq) ;
  for i in range (0, len(FT)) : 
     SUM = 0 ; 
     for t in range (0, len(S) - 1) :
        SUM = SUM + ( ( S[t]*math.cos(-2*math.pi*Freq[i]*Time[t]) ) + 1j*(S[t]*math.sin(-2*math.pi*Freq[i]*Time[t]) ) ) ;
     FT[i] = SUM ; 
  return FT ; 

#FUNCTIONS

def Delta(T) :
  Delta = [0]*int((T[1]- T[0])/T[2]) ;
  Delta[int(len(Delta)/2)+1] = 1 ; 
  return Delta ;

def Sine(T,f1,f2) :  
  Sine = [0]*int((T[1]- T[0])/T[2]) ;
  Time = np.arange(T[0], T[1], T[2]) ;
  for i in range (0, len(Sine) - 1) : 
     Sine[i] = math.sin(Time[i]*f1) + 2*math.sin(Time[i]*f2) ;
  return Sine; 

def SawTooth(T,f1,f2) :
   SawTooth = [0]*int((T[1]- T[0])/T[2]) ;
   Time = np.arange(T[0], T[1], T[2]) ;
   Controll_1 = 0; 
   for i in range (0, len(Sine) - 1) : 
     SawTooth[i] = f1*(Time[i] - Controll_1) ;
     if SawTooth[i] > 1 : 
        SawTooth[i] = 0 ;
        Controll_1 = Time[i]; 
   return SawTooth;
#---------------------

Time = np.array([0, 1*(10**6) , 10**3     ]) ;                  #HERE VARYING STARTING STOPPIN AND STEP
Freq = np.array([0, 5*(10**(-5)), 5*(10**(-8)) ]) ;     #HERE AS WELL IS THE TF SPACE
  
#Creation of the Signals

Delta = Delta(Time) ;
Sine = Sine (Time, 2*math.pi*(10**(-5)), 0 ) ;                  #HERE YOU CAN CHANGE FREQUENCY (also add another)
SawTooth = SawTooth(Time, 10**(-5), 0 ) ;

Frequency = np.arange(Freq[0], Freq[1], Freq[2]) ;              #This helps with the graphs
Tempo = np.arange(Time[0], Time[1], Time[2]) ;                  #This helps with the graphs

DATA = SawTooth ;                                                   #HERE TO CHANGE THE PLOT
FT = Fourier(DATA, Time, Freq) ;

fig = plt.figure(figsize=(20,12))
plt.plot(Tempo, DATA , 'r+')
plt.plot(Tempo, DATA )
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(20,12))
plt.plot(Frequency, np.abs(FT) , 'r+')
plt.plot(Frequency, np.abs(FT) )
plt.xlabel('Frequency 1/[year]')
plt.ylabel('Magnitude [not normalize]')
plt.grid(True)
plt.show()
