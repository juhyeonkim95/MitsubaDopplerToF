Doppler Time-of-Flight Renderer
===================================
## About
This is implementation of "Doppler Time-of-Rendering" which is submitted to on Mitsuba 0.5.

New integrator named `tofpath`, `tofantitheticpath`, `tofanalyticpath`  is added.
Each is used for uniform/stratified, antithetic sampling and analytic integration.

To compile, follow the original Mitsuba0.5's compliation guide.

Instead of default float precision, please use `config_double.py'.

Followings are some input parameters.

* `time` : exposure time in sec(default : 0.0015)
* `w_g` : illumination frequency in MHz (default : 30)
* `w_f` : sensor frequency in MHz (default : 30)
* `f_phase_offset` : sensor phase offset in radian (default : 0)
* `waveFunctionType` : modulation waveform (default : sinusoidal)
* `waveFunctionType` : modulation waveform (default : sinusoidal)
* `antitheticShifts` : antithetic shifts. Multiple input is available separated by underbar. (e.g 0.5 for single antithetic or 0.12_0.35 two antithetics ) (default : 0.5)
* `antitheticShiftsNumber` : antithetic shifts with equal interval. If this value is set, this is used instead of `antitheticShifts`. (default : 0)
* `low_frequency_component_only` : low pass filtering (default : True)
* `force_constant_attenuation` : force zero-order approximation used in [Heide, 2015]
* `timeSamplingMode` : time sampling mode. Only valid for `tofantitheticpath`. One of `antithetic` or `antithetic_mirror`.
* `spatialCorrelationMode` : spatial correlation mode. 
    * `none` : no correlation
    * `pixel` : only correlate camera ray
    * `position` : correlate intersection point
    * `direction` : correlate direction (this is equal to sampler correlation)
    * `selective` : select one of `position`, `direction` based on material property


We also included example configurations with result image in `config_example` folder.

The code is still not refactored yet. Also some of notations are different from Mitsuba3 version.
We will work on this later.
