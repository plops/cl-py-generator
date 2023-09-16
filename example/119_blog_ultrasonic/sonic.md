
fingerprint sensor by Qualcomm.

- i used to work on medical ultrasound imaging
- it would be very nice if a mass produced 2d array of ultrasonic transducers were available
- perhaps one could build a cheap 3D ultrasound

- the research group of Bernhard E. Boser developed the piezoelectric micromachined transducer

- here is one paper from this group. one of the authors, yipeng lu, works as senior engineer at qualcomm, when the paper was released.
```
NOVEMBER 2016 3-D Ultrasonic Fingerprint Sensor-on-a-Chip Hao-Yen Tang, Student Member, IEEE, Yipeng Lu, Xiaoyue Jiang, Student Member, IEEE, Eldwin J. Ng, Julius M. Tsai, David A. Horsley, Member, IEEE, and Bernhard E. Boser, Fellow, IEEE

Abstract— A fully integrated 3-D ultrasonic fingerprint sensoron-a-chip is presented. The device consists of a 110×56 piezoelectric micromachined ultrasonic transducer (PMUT) array bonded at the wafer level to custom readout electronics fabricated in a 180-nm CMOS process with a HV (24 V) transistor option. With the 24 V driving signal strength, the sensor consumes 280 µJ to image a 4.73 mm × 3.24 mm section of a fingerprint at a rate of 380 fps. A wakeup mode that detects the presence of a finger at 4 fps and dissipates 10 µW allows the proposed sensor to double as a power switch. The sensor is capable of imaging both the surface epidermal and subsurface dermal fingerprints and is insensitive to contaminations, including perspiration or oil. The 3-D imaging capability combined with the sensor’s sensitivity to the acoustic properties of the tissue translates into excellent robustness against spoofing attacks. 
```
- searching for these names brings up many patents by qualcomm:
  - Kostadin Dimitrov Djordjev 
  - Jessica Liu Strohmann
  - Nicholas Ian Buchan
  
- but here i will limit myself to information from the 2016 paper

- The paper contains the image of a fingerprint: *image will be inserted here*

- i found a heavily redacted document (
  https://s3.i-micronews.com/uploads/2019/07/SP19465-YOLE_Qualcomm-3D-Sonic-Sensor-Fingerprint_Sample.pdf
  ) by System Plus Consulting. They took a phone apart and imaged the
  sensor with electron microscopy. While the images don't allow to see
  details, the text makes it clear that Qualcomm 3D Sonic contains a
  PMUT sensor.

- i found a youtube video of a reporter showing a prototype phone with the fingerprint sensor:https://youtu.be/JeTm5sd8ktg?t=143
- there you can see the types of images that are possible with a
  larger sensor than what is described in the paper
  
- the PMUTs are a MEMS device and bonded to a CMOS ASIC (with 24V transistors)


- how the PMUT works
  - an AlN layer is sandwiched between electrodes
  - transmission: applying a voltage to the electrodes causes the
    membrane to buckle
  - reception: an incident pressure wave deflects the membrane and
    causes a charge across the transducer
  
  - the backside is in vaccuum the front side is coupled to the finger
    through a 250um layer of silicone. during transmission a wave is
    only directed towards the finger. it can't couple into the vaccuum

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
  

- for the device to be useful for proper 3D imaging it would be
  necessary to it is possible to get sample the voltage of every PMUT
  at 28MHz for a many samples to perform proper beamforming.
  
- this does not have to happen simultanously. sequential readout of
  every PMUT would already give interesting information.

- i wonder if it is possible to modify the devices found in mobile
  phones. i doubt it.

