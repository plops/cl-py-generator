Title: Timegating for 7 frames with a digital micro mirror device

# Introduction
- the paper "Diffraction-gated real-time ultrahigh-speed mapping
  photography" (see [2]) describes a method to capture 7 frames at
  very fast speed


# Method
- they use a digital micro mirror device (DMD). This is a MEMS device
  and its small mirrors can rapidly move between two orientations.

- the dmd is in the back focal plane of an imaging system
- it creates 7 copies of the input image (7 diffraction orders)
- all dmds move at the same time and have the same instantaneous tilt
- During the transition from an “all-off” pattern to an “all-on”
  pattern, the continuous and synchronous change of the tilt angle of
  each micromirror results in a time-varying phase profile, which
  induces sweeping of the diffraction envelope through the diffraction
  orders located in its moving trajectory.
  
- equation 1 describes the diffraction pattern of the dmd
$$
\begin{split}I_{\rm r}(x,t)&=2c_{\rm r}^{2}(1+\cos\,m\pi)\cdot \Bigg\{\Bigg[{\rm sinc}\left(\frac{x-f\sin(2\theta_{\rm b})}{\sqrt{2}\lambda f/w}\right)\\&\quad\cdot\!\sum_{m=-\infty}^{\infty}\delta\left(x-\frac{m\lambda f}{p}\right)\Bigg]\otimes{\rm sinc}\left(\frac{xL_{x^{\prime}}}{\lambda f}\right)\Bigg\}^{2}.\end{split}
$$
- it describes the the intensity of a point target observed at the
  intermediate image plane
- sinc related to the width of individual mirrors modulates the
  diffraction orders (a grating with periodicity given by mirror
  pitch). the argument of the sinc contains the instantaneous tilt
  angle \theta_b of the mirror. this is the only place where the time
  dependency enters the equation
- sinc related to the width of the device is convolved with every
  delta function of the grating 

- maybe the  term $cos m \pi$  outside the sum over m is a mistake.

- frame interval between adjacent orders: 0.21 us +/- 0.03 us (stddev)
- corresponds to 4.8 Million frames per second

# Applications
- obviously it is not a very versatile tool to be able to capture just
  7 consecutive frames at a high rate.
- the integration time is quite short, so how high light powers must
  be involved (or dmd mirror has to be cycled many times).
- some applications i would find interesting:
  - looking at the ablation behaviour of a pulse laser
  - spatial distribution of fiber modes of a multi-mode fiber that
    is being moved
  - spatial distribution of light coming from a photonic integrated
    chip that undergoes a change (e.g. super luminiscent diode or the
    output of a laser with modulated current in the gain section)


# References
- [1] fast controller for dmd https://ajile.ca/ajd-4500/

- [2] https://opg.optica.org/optica/fulltext.cfm?uri=optica-10-9-1223&id=538060
- [3] hacker news discussion https://news.ycombinator.com/item?id=37523314


# Final Post 

https://fourierlisp.blogspot.com/2023/09/time-gating-for-7-frames-with-digital.html
