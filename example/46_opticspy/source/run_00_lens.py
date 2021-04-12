import pandas as pd
import numpy as np
import opticspy.ray_tracing as opr
l=opr.lens.Lens(lens_name="Triplet", creator="XF")
l.FNO=5
l.lens_info()
l.add_wavelength(wl=(656.30    ))
l.add_wavelength(wl=(587.60    ))
l.add_wavelength(wl=(486.10    ))
l.add_field_YAN(angle=0)
l.add_field_YAN(angle=14)
l.add_field_YAN(angle=20)
l.list_wavelengths()
l.list_fields()
l.add_surface(number=1, radius=(1.0e+9), thickness=(1.0e+9), glass="air", output=True, STO=False)
l.add_surface(number=2, radius=(41.159090    ), thickness=(6.097550    ), glass="S-BSM18_ohara", output=True, STO=False)
l.add_surface(number=3, radius=(-957.83150    ), thickness=(9.3490    ), glass="air", output=True, STO=False)
l.add_surface(number=4, radius=(-51.320    ), thickness=(2.0320    ), glass="N-SF2_schott", output=True, STO=False)
l.add_surface(number=5, radius=(42.3780    ), thickness=(5.9960    ), glass="air", output=True, STO=False)
l.add_surface(number=6, radius=(1.0e+9), thickness=(4.0650    ), glass="air", output=True, STO=True)
l.add_surface(number=7, radius=(247.450    ), thickness=(6.0970    ), glass="S-BSM18_ohara", output=True, STO=False)
l.add_surface(number=8, radius=(-40.040    ), thickness=(85.590    ), glass="air", output=True, STO=False)
l.add_surface(number=9, radius=(1.0e+9), thickness=0, glass="air", output=True, STO=False)