
Title: How does the fingerprint sensor by Qualcomm work?

# Introduction

- i recently was looking at used phones and saw an interesting
  demonstration of the Samsung S22's fingerprint sensor.
  
- it took quite a while to find a reasonable description of how the
  device works. i think i found the group who initially developed the
  device and i think it is worthwhile to write down my findings

## My motivation
- i used to work on medical ultrasound imaging
- it would be very nice if a mass produced 2d array of ultrasonic transducers were available
- perhaps one could build a cheap 3D ultrasound

# The link to the initial Research Paper on the technology

- around 2015 the research group of Bernhard E. Boser developed the piezoelectric micromachined transducer

- reference [1]  is one paper from this group. one of the authors, yipeng lu, works as senior engineer at qualcomm, when the paper was released.
```
NOVEMBER 2016 3-D Ultrasonic Fingerprint Sensor-on-a-Chip Hao-Yen Tang, Student Member, IEEE, Yipeng Lu, Xiaoyue Jiang, Student Member, IEEE, Eldwin J. Ng, Julius M. Tsai, David A. Horsley, Member, IEEE, and Bernhard E. Boser, Fellow, IEEE

Abstract— A fully integrated 3-D ultrasonic fingerprint sensoron-a-chip is presented. The device consists of a 110×56 piezoelectric micromachined ultrasonic transducer (PMUT) array bonded at the wafer level to custom readout electronics fabricated in a 180-nm CMOS process with a HV (24 V) transistor option. With the 24 V driving signal strength, the sensor consumes 280 µJ to image a 4.73 mm × 3.24 mm section of a fingerprint at a rate of 380 fps. A wakeup mode that detects the presence of a finger at 4 fps and dissipates 10 µW allows the proposed sensor to double as a power switch. The sensor is capable of imaging both the surface epidermal and subsurface dermal fingerprints and is insensitive to contaminations, including perspiration or oil. The 3-D imaging capability combined with the sensor’s sensitivity to the acoustic properties of the tissue translates into excellent robustness against spoofing attacks. 
```
- searching for the following names brings up several more recent patents by qualcomm on the topic:
  - Kostadin Dimitrov Djordjev 
  - Jessica Liu Strohmann
  - Nicholas Ian Buchan
  
- however, in this post i will limit myself to information from the
  2016 paper

- The paper contains the image of a fingerprint: *image will be inserted here*

- i found a heavily redacted document (see [2]) by System Plus
  Consulting. They took a phone apart and imaged the sensor with
  electron microscopy. While the images are blurred and it is
  impossible to see details, the text makes it clear that Qualcomm 3D
  Sonic contains PMUT devices.

- i found a youtube video of a reporter showing a prototype phone with the fingerprint sensor [3]
- there you can see the types of images that are possible with a
  larger sensor than what is described in the paper *still image from the video will be inserted here*

# Construction of the Sensor
- the PMUTs are a MEMS device and bonded to a CMOS ASIC (with 24V transistors)

## How does the PMUT work?
  - an AlN layer is sandwiched between electrodes
  - transmission: applying a voltage to the electrodes causes the
    membrane to buckle
  - reception: an incident pressure wave deflects the membrane and
    causes a charge across the transducer
  
  - the backside is in vaccuum the front side is coupled to the finger
    through a 250um layer of silicone. during transmission a wave is
    only directed towards the finger. it can't couple into the vaccuum

## How is an array of PMUTs used as an imager?
- the array consists of 110 rows x 56 columns PMUTs with 43um x 58um
  pitch allowing sampling with approximately 500 dpi

- the PMUTs in a column can be read out simultaneously using 56 analog demodulators
- column sequential readout

- during transmit the column and 4 adjacent columns are excited with 3
  pulses at 14MHz. the amplitude of the pulses is 24V.
- adjacent columns are delayed to focus the beam into 120um line 250um
  above the PMUT array and cause a pressure wave with 15kPa amplitude
- without finger present 2.5kPa are received, resulting in 860uV
  amplitude on the PMUT
- with finger only 800Pa are receive, resulting in 340uV amplitude 

- a ring down time of 200ns between transmission and reception ensures
  a minimal detectable distance of 190um

- the selected column then reads the echoes
- a 4us delay ensures remaining echoes have dissipated before going to next column

- the image is formed by an analog method. the envelope of the
  received signal is sampled at a specific time.
  
- they also describe an image reconstruction technique based on
  digitization but mention that this is far too slow.
  
# Conclusion

## Positives
- the ultrasonic fingerprint sensor from qualcomm is a neat piece of
  engineering. it can rapidly (> 300 frames per second) image the
  surface of the finger (image A) as well as a layer deeper in the
  tissue (image B).
  
- the sensor works in total darkness and works through the phones
  display. fingerprints can be read reliably in wet conditions.

- apparently, the device uses so little power, that
  it can act as a power button.

## Questionable security improvement
- they claim the sensor is 3d and indeed they show image B and claim
  that comes from a deeper layer of the skin. it looks very similar to
  the image A from the surface of the finger, though.
- however, they employ a very primitive reconstruction method. i find
  it much more likely that their second image from "inside" the finger
  shows artifacts from the corrugations on the surface of the finger.
  
- it would be interesting to see if the device could discover a
  different structure behind in the depth (e.g. lines in image A and
  circles in image B).
- from what i understand it is certainly not proven that the 3D sensor
  (image B) improves the biometric security at all. i would not be
  surprised if the sensor can be tricked by a silicone cast of a
  finger.
  
## Are other applications possible?

- for the device to be useful for proper 3D imaging it would be
  necessary  to get sample the voltage of every PMUT
  at 28MHz for a many samples to perform proper beamforming.
  
- this does not have to happen simultanously. sequential readout of
  every PMUT would already give interesting information.

- i wonder if it is possible to modify the devices found in mobile
  phones. 
  
- as this is a high volume product that is heavily engineered to solve
  one particular problem at low power, i doubt this kind of
  modifaction is easy or possible.

# References
- [1] Tang, Hao-Yen, et al. "3-D ultrasonic fingerprint
  sensor-on-a-chip." IEEE Journal of Solid-State Circuits 51.11
  (2016): 2522-2533. https://ieeexplore.ieee.org/abstract/document/7579196
- [2] https://s3.i-micronews.com/uploads/2019/07/SP19465-YOLE_Qualcomm-3D-Sonic-Sensor-Fingerprint_Sample.pdf
- [3] https://youtu.be/JeTm5sd8ktg?t=143

# Final Post:

https://fourierlisp.blogspot.com/2023/09/how-does-qualcomms-fingerprint-sensor.html
