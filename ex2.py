#MAIO exercise2

import matplotlib.pyplot as plt
import math as M
import numpy as np


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



#Creation of the Signal
Signal = [0]*int((Time[1]- Time[0])/Time[2]);
Signal[int(len(Signal)/2)-1] = 1; 
Signal[int(len(Signal)/2)-1] = 1; 

fig = plt.figure(figsize=(20,12))
plt.plot(Signal,'r+')
plt.plot(Signal)
plt.grid(True)
plt.show()

Frequency = np.arange(Freq[0], Freq[1], Freq[2]) ;  #This helps with the graphs
FT = Fourier(Signal, Time, Freq) ;

fig = plt.figure(figsize=(20,12))
plt.plot(Frequency, np.power(FT,2) , 'r+')
plt.plot(Frequency, np.power(FT,2) )
plt.grid(True)
plt.show()
