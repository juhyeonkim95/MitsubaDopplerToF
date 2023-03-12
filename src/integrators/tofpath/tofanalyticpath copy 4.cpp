/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include "path_trace_parts.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("ToFPath tracer", "Average path length", EAverage);


class ToFAnalyticPathTracer : public MonteCarloIntegrator {
public:
    ToFAnalyticPathTracer(const Properties &props)
        : MonteCarloIntegrator(props) {
            
        m_needOffset = true;
        m_time = props.getFloat("time");
        m_illumination_modulation_frequency_mhz = props.getFloat("w_g", 30.0f);
        m_illumination_modulation_scale = props.getFloat("g_1", 0.5f);
        m_illumination_modulation_offset = props.getFloat("g_0", 0.5f);

        m_sensor_modulation_frequency_mhz = props.getFloat("w_f", 30.0f);
        m_sensor_modulation_scale = props.getFloat("f_1", 0.5f);
        m_sensor_modulation_offset = props.getFloat("f_0", 0.5f);
        m_sensor_modulation_phase_offset = props.getFloat("f_phase_offset", 0.0f);

        m_low_frequency_component_only = props.getBoolean("low_frequency_component_only", true);
        m_is_objects_transformed_for_time = props.getBoolean("is_object_transformed_for_time", false);
        m_force_constant_attenuation = props.getBoolean("force_constant_attenuation", false);
        
        m_antithetic_sampling_mode = props.getString("antitheticSamplingMode", "position");
        
        m_time_intervals = props.getInteger("timeIntervals", 1);
    }

    /// Unserialize from a binary data stream
    ToFAnalyticPathTracer(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }


    Float evalIntegratedModulationWeight(Float st, Float et, Float path_length, Float path_length_at_t, Float f_value_ratio_inc) const{
        Float w_g = 2 * M_PI * m_illumination_modulation_frequency_mhz * 1e6;
        Float w_f = 2 * M_PI * m_sensor_modulation_frequency_mhz * 1e6;
        Float temp = (2 * M_PI * m_illumination_modulation_frequency_mhz) / 300;
        Float w_delta = - temp * (path_length_at_t - path_length) / (et - st);
        Float phi = temp * path_length;

        Float a = (w_f - w_g - w_delta);
        Float b = phi;
        Float c = f_value_ratio_inc / (et - st);

        Float s1 = 0.5;
        if(std::abs(a) < 1e-3){
            Float BT = std::cos(b) * et;
            Float B0 = std::cos(b) * st;
            if(!m_force_constant_attenuation){
                BT += ( 0.5 * c * et * et * std::cos(b));
                B0 += ( 0.5 * c * st * st * std::cos(b));
            }
            return s1 / 2 * (BT - B0);
        } else {
            Float AT = std::sin(a * et + b) / a;
            Float A0 = std::sin(a * st + b) / a;

            if(!m_force_constant_attenuation){
                AT += ( c * et * std::sin(a * et + b) / a + c * std::cos(a * et + b) / (a * a));
                A0 += ( c * st * std::sin(a * st + b) / a + c * std::cos(a * st + b) / (a * a));
            }
            return s1 / 2 * (AT - A0);
        }
    }

    Spectrum accumulate_nee_Li(PathTracePart &path1, PathTracePart &path2) const{
        if(path1.path_pdf_nee_as_nee > 0){
            Float p11 = path1.path_pdf_as(path1);

            // NEE, next ray (bsdf / positional corr / sampler corr) MIS
            Float p_nee = path1.path_pdf_nee_as_nee;
            Float p_next_ray = path1.path_pdf_nee_as_next_ray;
            Float mis_nee_next_ray = miWeight(p_nee, p_next_ray, 2);

            Float path_pdf = p11 * p_nee;

            Spectrum f_value_ratio_em = (path2.neeEmitterValue * path2.path_throughput_nee) / (path1.neeEmitterValue * path1.path_throughput_nee);

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


    Spectrum accumulate_next_ray_Li(PathTracePart &path1, PathTracePart &path2) const{
        if(path1.hitEmitter){
            // Primal, antithetic MIS
            Float p11 = path1.path_pdf_as(path1);

            // NEE, next ray (bsdf / positional corr / sampler corr) MIS
            Float p_nee = path1.path_pdf_next_ray_as_nee;
            Float p_next_ray = path1.path_pdf_next_ray_as_next_ray;
            Float mis_nee_next_ray = miWeight(p_next_ray, p_nee, 2);

            Float path_pdf = p11;

            Spectrum f_value_ratio = (path2.hitEmitterValue * path2.path_throughput) / (path1.hitEmitterValue * path1.path_throughput);

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

    Spectrum Li_helper(const RayDifferential &r1,const RayDifferential &r2, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Spectrum Li(0.0f);
        RadianceQueryRecord rRec2 = rRec;

        // path 1
        RayDifferential ray1(r1);
        rRec.rayIntersect(ray1);
        PathTracePart path1(ray1, rRec, 0);

        // path 2
        RayDifferential ray2(r2);
        rRec2.rayIntersect(ray2);
        PathTracePart path2(ray2, rRec2, 1);
        bool use_positional_correlation = false;
        
        while(rRec.depth <= m_maxDepth || m_maxDepth < 0){
            /* Only continue if:
            1. The current path length is below the specifed maximum
            2. If 'strictNormals'=true, when the geometric and shading
                normals classify the incident direction to the same side */
                
            if (!path1.path_terminated && ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(path1.ray.d, path1.its.geoFrame.n)
                    * Frame::cosTheta(path1.its.wi) >= 0))) {
                path1.path_terminated = true;
            }

            if (!path2.path_terminated && ((rRec2.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(path2.ray.d, path2.its.geoFrame.n)
                    * Frame::cosTheta(path2.its.wi) >= 0))) {
                path2.path_terminated = true;
            }

            // check path is terminated
            if(path1.path_terminated && path2.path_terminated){
                break;
            }

            if(path1.path_terminated && !path2.path_terminated){
                path1.set_path_pdf_as(path1, 0.0);
                std::swap(path1, path2);
            }

            if(!path1.path_terminated && path2.path_terminated){
                path2.set_path_pdf_as(path2, 0.0);
            }

            // direct emission
            if(rRec.depth == 0 && !m_hideEmitters){
                Li += accumulate_next_ray_Li(path1, path2);
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */
            Point2 nee_sample = rRec.nextSample2D();
            path1.nee_trace(nee_sample);

            Float jacobian;

            // For nee, positional correlation is equal to sampler correlation.
            if(use_positional_correlation){
                path2.nee_trace(nee_sample);
                jacobian = 1;
            } else {
                path2.nee_trace(nee_sample);
                jacobian = path1.G_nee / path2.G_nee;
            }
            PathTracePart::calculate_nee_pdf(path1, path2, jacobian);

            
            // Accumulate nee Li
            Li += accumulate_nee_Li(path1, path2);

            if(rRec.depth + 1>=m_maxDepth){
                break;
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            Point2 bsdf_sample = rRec.nextSample2D();
            path1.get_next_ray_from_sample(bsdf_sample);

            use_positional_correlation = false;
            const BSDF *bsdf = path1.prev_its.getBSDF(path1.ray);
            Float its_roughness = bsdf->getRoughness(path1.prev_its, 0);
            
            if(strcmp(m_antithetic_sampling_mode.c_str(), "position") == 0){
                use_positional_correlation = true;
            }
            else if ((strcmp(m_antithetic_sampling_mode.c_str(), "adaptive") == 0)){
                use_positional_correlation = (its_roughness > 0.3);
            }

            if(path1.path_terminated){
                use_positional_correlation = false;
                path1.path_throughput *= 0.0;
                path1.set_path_pdf_as(path1, 0.0);
            }

            if(use_positional_correlation){
                path2.get_next_ray_from_its(path1.its);
                jacobian = 1;
            } else {
                path2.get_next_ray_from_sample(bsdf_sample);
                jacobian = path1.G / path2.G;
            }
            if(path2.path_terminated){
                path2.path_throughput *= 0.0;
                path2.set_path_pdf_as(path2, 0.0);
            }
            PathTracePart::calculate_next_ray_pdf(path1, path2, jacobian);
            
            path1.set_path_pdf_as(path1, path1.bsdfPdf * path1.G);
            PathTracePart::update_path_pdf(path1, path2, jacobian);

            // Accumulate next ray Li
            Li += accumulate_next_ray_Li(path1, path2);
            
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec2.type = RadianceQueryRecord::ERadianceNoEmission;
            
            rRec.depth++;
            rRec2.depth++;
        }
        return Li;
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &_rRec) const {
        Spectrum Li(0.0f);
        float k = 1.0 / ((float) m_time_intervals);
        _rRec.sampler->saveState();

        for(int i=0; i<m_time_intervals; i++){
            RadianceQueryRecord rRec = _rRec;
            rRec.sampler->loadSavedState();
            
            float start_time = i * k;
            float end_time = (i+1) * k;

            RayDifferential ray1;
            rRec.scene->getSensor()->sampleRayDifferential(
                ray1, _rRec.samplePos, _rRec.apertureSample, start_time);
            ray1.scaleDifferential(_rRec.diffScaleFactor);

            RayDifferential ray2;
            rRec.scene->getSensor()->sampleRayDifferential(
                ray2, _rRec.samplePos, _rRec.apertureSample, end_time);
            ray2.scaleDifferential(_rRec.diffScaleFactor);

            Li += Li_helper(ray1, ray2, rRec);
        }

        for(int i=0; i<m_time_intervals; i++){
            _rRec.sampler->advance();
        }
        return Li;
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "ToFAnalyticPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    Float m_illumination_modulation_frequency_mhz;
    Float m_illumination_modulation_scale;
    Float m_illumination_modulation_offset;
    Float m_sensor_modulation_frequency_mhz;
    Float m_sensor_modulation_scale;
    Float m_sensor_modulation_offset;
    Float m_sensor_modulation_phase_offset;
    
    bool m_low_frequency_component_only;
    bool m_is_objects_transformed_for_time;
    bool m_force_constant_attenuation;
    bool m_correlate_time_samples;
    std::string m_antithetic_sampling_mode;

    Float m_time;
    int m_time_intervals;
};

MTS_IMPLEMENT_CLASS_S(ToFAnalyticPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ToFAnalyticPathTracer, "ToF analytic path tracer");
MTS_NAMESPACE_END
