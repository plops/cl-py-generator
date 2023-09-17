- https://opg.optica.org/optica/fulltext.cfm?uri=optica-10-9-1223&id=538060
- https://news.ycombinator.com/item?id=37523314
- fast controller for dmd https://ajile.ca/ajd-4500/

- the dmd is in the back focal plane of an imaging system
- it creates 7 copies of the input image (7 diffraction orders)
- all dmds move at the same time and have the same instantaneous tilt
- During the transition from an “all-off” pattern to an “all-on”
  pattern, the continuous and synchronous change of the tilt angle of
  each micromirror results in a time-varying phase profile (Supplement
  1, Fig. S4c), which induces sweeping of the diffraction envelope
  through the diffraction orders located in its moving trajectory.
  
- equation 1 is describes the diffraction pattern of the dmd
- it describes the the intensity of a point target observed at the
  intermediate image plane
- sinc related to the width of individual mirrors modulates the
  diffraction orders (a grating with periodicity given by mirror
  pitch). the argument of the sinc contains the instantaneous tilt
  angle \theta_b of the mirror. this is the only place where the time
  dependency enters the equation
- sinc related to the width of the device is convolved with every
  delta function of the grating 

- why does the term $cos m \pi$ occur outside the sum of m?

- frame interval between adjacent orders: 0.21 us +/- 0.03 us (stddev)
- corresponds to 4.8Mfps 
