Doppler Time-of-Flight Renderer
===================================
## About
This is Mitsuba0.5 implementation of "Doppler Time-of-Rendering" submitted to SIGGRAPH Asia 2023.
Please also check Mitsuba3 implementation at []here.

New integrator named `tofpath`, `tofantitheticpath`, `tofanalyticpath`  is added.
Each is used for uniform/stratified, antithetic sampling and analytic integration.

To compile, follow the original Mitsuba0.5's compliation guide.

Instead of default float precision, please use `config_double.py'.

Followings are some input parameters.

### ToF Related

* `time` : Exposure time in sec. (default : 0.0015)
* `w_g` : Illumination frequency in MHz. (default : 30)
* `w_s` : Sensor frequency in MHz. (default : 30)
* `sensor_phase_offset` : Sensor phase offset in radian. (default : 0)
* `wave_function_type` : Modulation waveform. Refer following table for exact configuration. (default : sinusoidal)

| `wave_function_type` | Sensor Modulation | Light Modulation | Low Pass Filtered |
|-------------|-------------------|------------------|-------------------|
| `sinusoidal`  | sinusoidal        | sinusoidal       | sinusoidal        |
| `rectangular` | rectangular       | rectangular      | triangular        |
| `triangular`  | triangular        | triangular       | Corr(tri, tri)    |
| `trapezoidal` | trapezoidal       | delta            | trapezoidal       |

* `low_frequency_component_only` : Whether to use low pass filtering for modulation functions. (default : true)


### Sampling Related
* `time_sampling_mode` : Times sampling method.
    * `uniform` : uniform sampling
    * `stratified` : stratified sampling
    * `antithetic` : shifted antithetic sampling
    * `antithetic_mirror` : mirrored antithetic sampling
    * `analytic` : analytic integration (biased)

* `antithetic_shifts` : User defined antithetic shifts. Multiple input is available separated by underbar. (e.g 0.5 for single antithetic sample or 0.12_0.35 two antithetic samples) (default : 0.5 for `antithetic`, 0.0 for `antithetic_mirror`)
* `antithetic_shifts_number` : Numer of antithetic shifts with equal interval. If this value is set, this is used instead of `antithetic_shifts`. (default : 0)
* `m_use_full_time_stratification` : Whether to use full stratification over time. If set to `true`, it works differently by `time_sampling_mode`. (default : False)
    * `stratified` : correlated randomly over different stratum (Fig.8-(b) in the main paper)
    * `antithetic` : use stratification for primal sample (Fig.8-(e) in the main paper)
    * `antithetic_mirror` : use stratification for primal sample (Fig.8-(d) in the main paper)

* `spatial_correlation_mode` : Spatial correlation methods. Note that methods start with `ray` explicitly correlate two paths in ray-by-ray style.
    * `none` : no correlation
    * `pixel` : repeat camera ray (use same pixel coordinate) between multiple rays
    * `sampler` : repeat sampler between multiple rays
    * `ray_position` : correlate intersection point between two rays
    * `ray_sampler` : repeat sampler between two rays
    * `ray_selective` : select one of `ray_position`, `ray_sampler` based on material property

* `force_constant_attenuation` : Whether to force constant attenuation as [Heide, 2015] did. `true` is using zero-order Taylor approximation while `false` is using first-order Taylor approximation. Only used for `analytic`. (default : false)
* `primal_antithetic_mis_power` : MIS power for primal and antithetic sample. Only used for other than `analytic`. Refer Sec 4.1 in supplementary material for details. (default : 1.0)

We also included example configurations with result image in `config_example` folder.