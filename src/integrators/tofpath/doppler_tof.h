#include "waveform_utils.h"
#include "path_trace_parts.h"

// Helper class to make doppler_tof_path.cpp shorter
class DopplerToF {
public:
    DopplerToF() {};
    DopplerToF(const Properties &props) {
        m_time = props.getFloat("time", 0.0015f);    // exposure time

        // modulation function frequency / scale / offset
        m_illumination_modulation_frequency_mhz = props.getFloat("w_g", 30.0f);
        m_illumination_modulation_scale = props.getFloat("g_1", 0.5f);
        m_illumination_modulation_offset = props.getFloat("g_0", 0.5f);
        m_sensor_modulation_frequency_mhz = props.getFloat("w_s", 30.0f);
        m_sensor_modulation_phase_offset = props.getFloat("sensor_phase_offset", 0.0f);
        
        // syntactic sugar
        if(props.hasProperty("hetero_offset")){
            m_sensor_modulation_phase_offset = props.getFloat("hetero_offset", 0.0) * 2 * M_PI;
        }
        if(props.hasProperty("hetero_frequency")){
            Float w_d = 1 / m_time * props.getFloat("hetero_frequency", 0.0);
            m_sensor_modulation_frequency_mhz = m_illumination_modulation_frequency_mhz + w_d * 1e-6;
        }

        // modulation function waveform
        std::string wave_function_type_str = props.getString("wave_function_type", "sinusoidal");

        if(strcmp(wave_function_type_str.c_str(), "sinusoidal") == 0){
            m_wave_function_type = WAVE_TYPE_SINUSOIDAL;
        }
        else if(strcmp(wave_function_type_str.c_str(), "rectangular") == 0){
            m_wave_function_type = WAVE_TYPE_RECTANGULAR;
        }
        else if(strcmp(wave_function_type_str.c_str(), "triangular") == 0){
            m_wave_function_type = WAVE_TYPE_TRIANGULAR;
        }
        else if(strcmp(wave_function_type_str.c_str(), "trapezoidal") == 0){
            m_wave_function_type = WAVE_TYPE_TRAPEZOIDAL;
        }
        
        m_low_srequency_component_only = props.getBoolean("low_srequency_component_only", true);
        m_force_constant_attenuation = props.getBoolean("force_constant_attenuation", false);
        m_primal_antithetic_mis_power = props.getFloat("primal_antithetic_mis_power", 1.0f);
    };

    Float evalIntegratedModulationWeight(Float st, Float et, Float path_length, Float path_length_at_t, Float f_value_ratio_inc) const{
        Float w_g = 2 * M_PI * m_illumination_modulation_frequency_mhz * 1e6;
        Float w_s = 2 * M_PI * m_sensor_modulation_frequency_mhz * 1e6;
        Float temp = (2 * M_PI * m_illumination_modulation_frequency_mhz) / 300;
        Float w_delta = - temp * (path_length_at_t - path_length) / (et - st);
        Float phi = temp * path_length;

        Float a = (w_s - w_g - w_delta);
        Float b = phi;
        Float c = m_force_constant_attenuation? 0 : f_value_ratio_inc / (et - st);
        
        Float A = evalModulationFunctionValueLowPassIntegrated(m_wave_function_type, a, b, c, et);
        Float B = evalModulationFunctionValueLowPassIntegrated(m_wave_function_type, a, b, c, st);
        return m_illumination_modulation_scale / 2 * (A - B);
    }

    Float evalModulationWeight(Float &ray_time, Float &path_length) const{
        Float w_g = 2 * M_PI * m_illumination_modulation_frequency_mhz * 1e6;
        Float w_s = 2 * M_PI * m_sensor_modulation_frequency_mhz * 1e6;
        Float w_d = 2 * M_PI * (m_sensor_modulation_frequency_mhz - m_illumination_modulation_frequency_mhz) * 1e6;
        Float phi = (2 * M_PI * m_illumination_modulation_frequency_mhz) / 300 * path_length;

        if(m_low_srequency_component_only){
            Float t = w_d * ray_time + m_sensor_modulation_phase_offset + phi;
            Float sg_t = 0.5 * m_illumination_modulation_scale * evalModulationFunctionValueLowPass(m_wave_function_type, t);
            return sg_t;
        } 
        
        Float t1 = w_g * ray_time - phi;
        Float t2 = w_s * ray_time + m_sensor_modulation_phase_offset;
        Float g_t = m_illumination_modulation_scale * evalModulationFunctionValue(m_wave_function_type, t1) + m_illumination_modulation_offset;
        Float s_t = evalModulationFunctionValue(m_wave_function_type, t2);
        return s_t * g_t;
    }

    Spectrum accumulate_nee_Li_analytic(PathTracePart &path1, PathTracePart &path2, Float jacobian) const{
        if(path1.path_pdf_nee_as_nee > 0){
            Float p11 = path1.path_pdf_as(path1);

            // NEE, next ray (bsdf / positional corr / sampler corr) MIS
            Float p_nee = path1.path_pdf_nee_as_nee;
            Float p_next_ray = path1.path_pdf_nee_as_next_ray;
            Float mis_nee_next_ray = miWeight(p_nee, p_next_ray, 2);

            Float path_pdf = p11 * p_nee;

            Spectrum f_value_ratio_em = (path2.neeEmitterValue * path2.path_throughput_nee) / (path1.neeEmitterValue * path1.path_throughput_nee);
            f_value_ratio_em *= jacobian;

            Float integrated_modulation_x = evalIntegratedModulationWeight(path1.ray.time, path2.ray.time, 
                path1.em_path_length, path2.em_path_length, f_value_ratio_em[0] - 1);
            Float integrated_modulation_y = evalIntegratedModulationWeight(path1.ray.time, path2.ray.time, 
                path1.em_path_length, path2.em_path_length, f_value_ratio_em[1] - 1);
            Float integrated_modulation_z = evalIntegratedModulationWeight(path1.ray.time, path2.ray.time, 
                path1.em_path_length, path2.em_path_length, f_value_ratio_em[2] - 1);
            Spectrum integrated_modulation = Color3(integrated_modulation_x, integrated_modulation_y, integrated_modulation_z);
            return integrated_modulation * path1.neeEmitterValue * path1.path_throughput_nee / path_pdf * mis_nee_next_ray;
        }
        return Spectrum(0.0);
    }


    Spectrum accumulate_next_ray_Li_analytic(PathTracePart &path1, PathTracePart &path2, Float jacobian) const{
        if(path1.hitEmitter){
            // Primal, antithetic MIS
            Float p11 = path1.path_pdf_as(path1);

            // NEE, next ray (bsdf / positional corr / sampler corr) MIS
            Float p_nee = path1.path_pdf_next_ray_as_nee;
            Float p_next_ray = path1.path_pdf_next_ray_as_next_ray;
            Float mis_nee_next_ray = miWeight(p_next_ray, p_nee, 2);

            Float path_pdf = p11;

            Spectrum f_value_ratio = (path2.hitEmitterValue * path2.path_throughput) / (path1.hitEmitterValue * path1.path_throughput);
            f_value_ratio *= jacobian;

            Float integrated_modulation_x = evalIntegratedModulationWeight(path1.ray.time, path2.ray.time, 
                path1.path_length, path2.path_length, f_value_ratio[0] - 1);
            Float integrated_modulation_y = evalIntegratedModulationWeight(path1.ray.time, path2.ray.time, 
                path1.path_length, path2.path_length, f_value_ratio[1] - 1);
            Float integrated_modulation_z = evalIntegratedModulationWeight(path1.ray.time, path2.ray.time, 
                path1.path_length, path2.path_length, f_value_ratio[2] - 1);
            Spectrum integrated_modulation = Color3(integrated_modulation_x, integrated_modulation_y, integrated_modulation_z);

            return integrated_modulation * path1.hitEmitterValue * path1.path_throughput / path_pdf * mis_nee_next_ray;
        }
        return Spectrum(0.0);
    }

    Spectrum accumulate_nee_Li(PathTracePart &path1, PathTracePart &path2) const{
        if(path1.path_pdf_nee_as_nee > 0){
            // Primal, antithetic MIS
            Float p11 = path1.path_pdf_as(path1);
            Float p12 = path1.path_pdf_as(path2);
            Float mis_primal_antithetic = miWeight(p11, p12, m_primal_antithetic_mis_power);

            // NEE, next ray (bsdf / positional corr / sampler corr) MIS
            Float p_nee = path1.path_pdf_nee_as_nee;
            Float p_next_ray = path1.path_pdf_nee_as_next_ray;
            Float mis_nee_next_ray = miWeight(p_nee, p_next_ray, 2);

            Float path_pdf = p11 * p_nee;
            Float modulation = evalModulationWeight(path1.ray.time, path1.em_path_length);
            return modulation * path1.path_throughput_nee * path1.neeEmitterValue / path_pdf * mis_primal_antithetic * mis_nee_next_ray;
        }
        return Spectrum(0.0);
    }

    Spectrum accumulate_next_ray_Li(PathTracePart &path1, PathTracePart &path2) const{
        if(path1.hitEmitter){
            // Primal, antithetic MIS
            Float p11 = path1.path_pdf_as(path1);
            Float p12 = path1.path_pdf_as(path2);
            Float mis_primal_antithetic = miWeight(p11, p12, m_primal_antithetic_mis_power);
    
            // NEE, next ray (bsdf / positional corr / sampler corr) MIS
            Float p_nee = path1.path_pdf_next_ray_as_nee;
            Float p_next_ray = path1.path_pdf_next_ray_as_next_ray;
            Float mis_nee_next_ray = miWeight(p_next_ray, p_nee, 2);

            Float path_pdf = p11;
            Float modulation = evalModulationWeight(path1.ray.time, path1.path_length);
            return modulation * path1.path_throughput * path1.hitEmitterValue / path_pdf * mis_primal_antithetic * mis_nee_next_ray;
        }
        return Spectrum(0.0);
    }

protected:
    Float m_illumination_modulation_frequency_mhz;
    Float m_illumination_modulation_scale;
    Float m_illumination_modulation_offset;
    Float m_sensor_modulation_frequency_mhz;
    Float m_sensor_modulation_scale;
    Float m_sensor_modulation_offset;
    Float m_sensor_modulation_phase_offset;
    Float m_time;
    
    bool m_low_srequency_component_only;
    bool m_force_constant_attenuation;
    float m_primal_antithetic_mis_power;

    EWaveformType m_wave_function_type;
};