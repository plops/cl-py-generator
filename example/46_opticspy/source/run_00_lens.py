import pandas as pd
import numpy as np
import opticspy.ray_tracing as opr
l=opr.lens.Lens(lens_name="Triplet", creator="XF")
l.FNO=5
l.lens_info()
l.add_wavelength(wl=(656.30    ))
l.add_wavelength(wl=(587.60    ))
l.add_wavelength(wl=(486.10    ))
l.list_wavelengths()