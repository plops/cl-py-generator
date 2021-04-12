import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import numpy as np
from opticspy.ray_tracing import *
l=lens.Lens(lens_name="Triplet", creator="XF")
l.FNO=5
l.lens_info()
l.add_wavelength(wl=(656.30    ))
l.add_wavelength(wl=(587.60    ))
l.add_wavelength(wl=(486.10    ))
l.list_wavelengths()