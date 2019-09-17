#MAIO exercise2

import matplotlib.pyplot as plt
import math 
import numpy as np
from scipy import signal



def Fourier(S, T, F) :

  #[Signal]    #S is the data set
  #[Time]      #T is an array with start time,      finish time,      time intervall
  #[Frequency] #F is an array with start frequency, finish frequency, frequency intervall

  Time = np.arange(T[0], T[1], T[2]) ;
  Freq = np.arange(F[0], F[1], F[2]) ;
  FT = [0]*len(Freq) ;
  for i in range (0, len(FT)-1) : 
     SUM = 0 ; 
     for t in range (0, len(S) -1) :
        SUM = SUM + ( S[t]*math.cos(-2*math.pi*Freq[i]*Time[t])*T[2] ) ;
     FT[i] = SUM ; 
  return FT ; 


Time = np.array([0, 0.6*(10**6) , 10**3     ]) ; 
Freq = np.array([0, 5*(10**(-5)), 5*10**(-8)]) ;


#Creation of the delta Signal
Signal = [0]*int((Time[1]- Time[0])/Time[2]);
Signal[int(len(Signal)/2)] = 1; 
fig = plt.figure(figsize=(12,7))
##plt.plot(Signal,'r+')
plt.plot(Signal)
plt.grid(True)
plt.show()

Frequency = np.arange(Freq[0], Freq[1], Freq[2]) ;  #This helps with the graphs
FT = Fourier(Signal, Time, Freq) ;

fig = plt.figure(figsize=(12,7))
##plt.plot(Frequency, np.power(FT,2) , 'r+')
plt.plot(Frequency, np.power(FT,2) )
plt.grid(True)
plt.show() 


#Creation of the Sinusoidal Signal

Time = np.array([0.0, 10**6 , 10**3]) ;
N = 10.0**3
A = 1.0
f = 10.0
T = 1.0 / N
Time = np.array([0, 10**6 , 10**3     ]) ; 
x = np.linspace(0.0,N*T,N)
Signal = A*np.sin(f * 2.0 * np.pi*x)

fig = plt.figure(figsize=(12,7))
##plt.plot(Signal,'r+')
plt.plot(Signal)
plt.grid(True)
plt.show()

FT = Fourier(Signal, Time, Freq) ;

fig = plt.figure(figsize=(12,7))
##plt.plot(Frequency, np.power(FT,2) , 'r+')
plt.plot(Frequency, np.power(FT,2) )
plt.grid(True)
plt.show() 


#Creation of the Sawtooth Signal


A = 1.0 #this might work better for ex5
f = 10.0
Time = np.array([0.0, 10**6 , 10**3]) ;
T = np.arange(0.0,10**6,10**3)
x = np.linspace(0, 1, 1000)
Signal = A*signal.sawtooth(2*f * np.pi  * x + np.pi)
plt.plot(T, Signal)
plt.grid(True)
plt.show() 
FT = Fourier(Signal, Time, Freq) ;

fig = plt.figure(figsize=(12,7))
##plt.plot(Frequency, np.power(FT,2) , 'r+')
plt.plot(Frequency, np.power(FT,2) )
plt.grid(True)
plt.show() 

