# BeWoDa

## Input data
* Temperature (inside/outside) : {-20 ; 50} $\pm$ 5 °C  ==> **14 classes**
* Humidity (inside/outside) :  {0 ; 100} $\pm$ 5 %  ==> **20 classes**
* Atmospheric pressure : {850 ; 1050} $\pm$ 50 hPa  ==> **4 classes**
* CO$_2$ concentration : {200 ; 1200} $\pm$ 200 ppm  ==> **5 classes**
* Emotions : {'neutral', 'happy', 'sad', 'surprise', 'anger'}  ==> **5 classes**
* Body position : 18*2 values

Total state (without Body Positions) : $14\times 20 \times 4 \times 5 \times 5 = 28000$.

## States
$$
    s = s_r + s_w + s_h
$$
With $s_r$ the state of Yokobo (position of motors and light colour), $s_w$ the state of the world (temperature, humidity, AP and CO$_2$ level) and $s_h$ the state of the person (body positions and emotions).

