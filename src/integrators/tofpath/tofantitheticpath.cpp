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
#include "../../shapes/instance.h"
#include "path_trace_parts.h"
#include "parse_helper.h"


MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("ToFAntitheticPath tracer", "Average path length", EAverage);

/*! \plugin{tofpath}{ToFPath tracer}
 * \order{2}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *         A value of \code{1} will only render directly visible light sources.
 *         \code{2} will lead to single-bounce (direct-only) illumination,
 *         and so on. \default{\code{-1}}
 *     }
 *     \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *        which the implementation will start to use the ``russian roulette''
 *        path termination criterion. \default{\code{5}}
 *     }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See the description below
 *        for details.\default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 *
 * This integrator implements a basic path tracer and is a \emph{good default choice}
 * when there is no strong reason to prefer another method.
 *
 * To use the path tracer appropriately, it is instructive to know roughly how
 * it works: its main operation is to trace many light paths using \emph{random walks}
 * starting from the sensor. A single random walk is shown below, which entails
 * casting a ray associated with a pixel in the output image and searching for
 * the first visible intersection. A new direction is then chosen at the intersection,
 * and the ray-casting step repeats over and over again (until one of several
 * stopping criteria applies).
 * \begin{center}
 * \includegraphics[width=.7\textwidth]{images/integrator_path_figure.pdf}
 * \end{center}
 * At every intersection, the path tracer tries to create a connection to
 * the light source in an attempt to find a \emph{complete} path along which
 * light can flow from the emitter to the sensor. This of course only works
 * when there is no occluding object between the intersection and the emitter.
 *
 * This directly translates into a category of scenes where
 * a path tracer can be expected to produce reasonable results: this is the case
 * when the emitters are easily ``accessible'' by the contents of the scene. For instance,
 * an interior scene that is lit by an area light will be considerably harder
 * to render when this area light is inside a glass enclosure (which
 * effectively counts as an occluder).
 *
 * Like the \pluginref{direct} plugin, the path tracer internally relies on multiple importance
 * sampling to combine BSDF and emitter samples. The main difference in comparison
 * to the former plugin is that it considers light paths of arbitrary length to compute
 * both direct and indirect illumination.
 *
 * For good results, combine the path tracer with one of the
 * low-discrepancy sample generators (i.e. \pluginref{ldsampler},
 * \pluginref{halton}, or \pluginref{sobol}).
 *
 * \paragraph{Strict normals:}\label{sec:strictnormals}
 * Triangle meshes often rely on interpolated shading normals
 * to suppress the inherently faceted appearance of the underlying geometry. These
 * ``fake'' normals are not without problems, however. They can lead to paradoxical
 * situations where a light ray impinges on an object from a direction that is classified as ``outside''
 * according to the shading normal, and ``inside'' according to the true geometric normal.
 *
 * The \code{strictNormals}
 * parameter specifies the intended behavior when such cases arise. The default (\code{false}, i.e. ``carry on'')
 * gives precedence to information given by the shading normal and considers such light paths to be valid.
 * This can theoretically cause light ``leaks'' through boundaries, but it is not much of a problem in practice.
 *
 * When set to \code{true}, the path tracer detects inconsistencies and ignores these paths. When objects
 * are poorly tesselated, this latter option may cause them to lose a significant amount of the incident
 * radiation (or, in other words, they will look dark).
 *
 * The bidirectional integrators in Mitsuba (\pluginref{bdpt}, \pluginref{pssmlt}, \pluginref{mlt} ...)
 * implicitly have \code{strictNormals} set to \code{true}. Hence, another use of this parameter
 * is to match renderings created by these methods.
 *
 * \remarks{
 *    \item This integrator does not handle participating media
 *    \item This integrator has poor convergence properties when rendering
 *    caustics and similar effects. In this case, \pluginref{bdpt} or
 *    one of the photon mappers may be preferable.
 * }
 */


class ToFAntitheticPathTracer : public MonteCarloIntegrator {
public:
    ToFAntitheticPathTracer(const Properties &props)
        : MonteCarloIntegrator(props) {
        m_needOffset = true; // output can be negative, so add offset to make it positive
        m_time = props.getFloat("time", 0.0015f);    // exposure time

        // Modulation Function Related
        m_illumination_modulation_frequency_mhz = props.getFloat("w_g", 30.0f);
        m_illumination_modulation_scale = props.getFloat("g_1", 0.5f);
        m_illumination_modulation_offset = props.getFloat("g_0", 0.5f);
        m_sensor_modulation_frequency_mhz = props.getFloat("w_f", 30.0f);
        m_sensor_modulation_phase_offset = props.getFloat("f_phase_offset", 0.0f);
        std::string wave_function_type_str = props.getString("waveFunctionType", "sinusoidal");
        
        if(strcmp(wave_function_type_str.c_str(), "sinusoidal") == 0){
            m_wave_function_type = WAVE_TYPE_SINUSOIDAL;
        }
        if(strcmp(wave_function_type_str.c_str(), "rectangular") == 0){
            m_wave_function_type = WAVE_TYPE_RECTANGULAR;
        }
        if(strcmp(wave_function_type_str.c_str(), "triangular") == 0){
            m_wave_function_type = WAVE_TYPE_TRIANGULAR;
        }
        if(strcmp(wave_function_type_str.c_str(), "sawtooth") == 0){
            m_wave_function_type = WAVE_TYPE_SAWTOOTH;
        }
        if(strcmp(wave_function_type_str.c_str(), "trapezoidal") == 0){
            m_wave_function_type = WAVE_TYPE_TRAPEZOIDAL;
        }

        int antithetic_shift_number = props.getInteger("antitheticShiftsNumber", 0);
        if(antithetic_shift_number > 0){
            m_antithetic_shifts.clear();
            for(int i=1; i<antithetic_shift_number; i++){
                float t = 1.0 / ((float) antithetic_shift_number) * i;
                m_antithetic_shifts.push_back(t);
                std::cout << t << std::endl;
            }
        } else {    
            std::string antithetic_shift_string = props.getString("antitheticShifts", "0.5");
            m_antithetic_shifts = parse_float_array_from_string(antithetic_shift_string);
        }

        m_low_frequency_component_only = props.getBoolean("low_frequency_component_only", false);
        m_force_constant_attenuation = props.getBoolean("force_constant_attenuation", false); 

        m_time_sampling_mode = props.getString("timeSamplingMode", "uniform");
        m_spatial_correlation_mode = props.getString("spatialCorrelationMode", "none");
        m_primal_antithetic_mis_power = props.getFloat("primalAntitheticMISPower", 1.0f);
    }

    /// Unserialize from a binary data stream
    ToFAntitheticPathTracer(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    Float evalModulationFunctionValue(Float _t) const{
        Float t = std::fmod(_t, 2 * M_PI);
        switch(m_wave_function_type){
            case WAVE_TYPE_SINUSOIDAL: return std::cos(t);
            case WAVE_TYPE_RECTANGULAR: return std::abs(t-M_PI) > 0.5 * M_PI ? 1 : -1; //return dr::select(, 1, -1); //return dr::sign(dr::cos(t));
            case WAVE_TYPE_TRIANGULAR: return t < M_PI ? 1 - 2 * t / M_PI : -3 + 2 * t / M_PI;
        }
        return std::cos(t);
    }

    Float evalModulationFunctionValueLowPass(Float _t) const{
        Float t = std::fmod(_t, 2 * M_PI);
        switch(m_wave_function_type){
            case WAVE_TYPE_SINUSOIDAL: return std::cos(t);
            case WAVE_TYPE_RECTANGULAR: {
                Float a = t / M_PI;
                Float b = 2 - a;
                Float c = std::min(a, b);
                return 2 - 4 * c;
            }
            case WAVE_TYPE_TRIANGULAR: {    
                Float a = t / M_PI;
                Float b = 2 - a;
                Float c = std::min(a, b);
                return (4 * c * c * c - 6 * c * c + 1) * 2.0 / 3.0;
            }
            case WAVE_TYPE_TRAPEZOIDAL: {
                Float a = t / M_PI;
                Float b = 2 - a;
                Float c = std::min(a, b);
                Float r = 2 - 4 * c;
                return math::clamp(2.0 * r, -2.0, 2.0);
            }
        }
        return std::cos(t);
    }

    Float evalModulationWeight(Float &ray_time, Float &path_length) const{
        Float w_g = 2 * M_PI * m_illumination_modulation_frequency_mhz * 1e6;
        Float w_f = 2 * M_PI * m_sensor_modulation_frequency_mhz * 1e6;
        Float w_d = 2 * M_PI * (m_sensor_modulation_frequency_mhz - m_illumination_modulation_frequency_mhz) * 1e6;
        Float phi = (2 * M_PI * m_illumination_modulation_frequency_mhz) / 300 * path_length;

        if(m_low_frequency_component_only){
            Float t = w_d * ray_time + m_sensor_modulation_phase_offset + phi;
            Float fg_t = 0.5 * m_illumination_modulation_scale * evalModulationFunctionValueLowPass(t);
            return fg_t;
        } 
        
        Float t1 = w_g * ray_time - phi;
        Float t2 = w_f * ray_time + m_sensor_modulation_phase_offset;
        Float g_t = m_illumination_modulation_scale * evalModulationFunctionValue(t1) + m_illumination_modulation_offset;
        Float f_t = evalModulationFunctionValue(t2);
        return f_t * g_t;
    }

    Spectrum accumulate_nee_Li(PathTracePart &path1, PathTracePart &path2, bool debug=false) const{
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

    // Trace ray with single sample
    Spectrum Li_with_single_sample(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        Float path_length = 0;
        path_length += its.t;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
            {
                Float modulation_weight = evalModulationWeight(ray.time, path_length);
                Li += modulation_weight * throughput * its.Le(-ray.d);
            }

            /* Include radiance from a subsurface scattering model if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
                    * Frame::cosTheta(its.wi) >= 0)) {

                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                            || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;

                        /* Weight using the power heuristic */
                        Float weight = miWeight(dRec.pdf, bsdfPdf);

                        Float em_path_length = path_length + dRec.dist;

                        Float em_modulation_weight = evalModulationWeight(ray.time, em_path_length);

                        Li += throughput * value * bsdfVal * weight * em_modulation_weight;
                    }
                }
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;

            scattered |= bRec.sampledType != BSDF::ENull;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            bool hitEmitter = false;
            Spectrum value;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);
            if (scene->rayIntersect(ray, its)) {
                /* Intersected something - check if it was a luminaire */
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = scene->getEnvironmentEmitter();

                if (env) {
                    if (m_hideEmitters && !scattered)
                        break;

                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray))
                        break;
                    hitEmitter = true;
                } else {
                    break;
                }
            }

            path_length += its.t;
            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;

                Float modulation_weight = evalModulationWeight(ray.time, path_length);
                
                Li += throughput * value * miWeight(bsdfPdf, lumPdf) * modulation_weight;
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }
        }

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;
        // Li = Spectrum(-1.0);

        return Li;
    }

    // Trace ray with paired sample (total 2)
    Spectrum Li_with_paired_sample(
            const RayDifferential &r1,
            const RayDifferential &r2, 
            RadianceQueryRecord &rRec
        ) const {
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

        Point3 light_origin = ray1.o;
        while(rRec.depth <= m_maxDepth || m_maxDepth < 0){
            bool use_positional_correlation = false;
            if(strcmp(m_spatial_correlation_mode.c_str(), "position") == 0){
                use_positional_correlation = true;
            }
            if(path1.its.isValid()){
                const BSDF *bsdf = path1.its.getBSDF(path1.ray);
                Float its_roughness = bsdf->getRoughness(path1.its, 0);
                // if(its_roughness == 0.0){
                //     use_positional_correlation = false;
                // }

                its_roughness = std::min(its_roughness, 2.0);

                if(strcmp(m_spatial_correlation_mode.c_str(), "selective") == 0){
                    if(its_roughness > 0.5){
                        use_positional_correlation = true;
                    }
                }

                if(strcmp(m_spatial_correlation_mode.c_str(), "adaptive") == 0){
                    if(rRec.depth == 1){
                        Float position_correlation_probability = rRec.use_positional_correlation_probability;
                        // position_correlation_probability = position_correlation_probability * its_roughness;
                        // position_correlation_probability = position_correlation_probability / 0.1;
                        // position_correlation_probability = std::power(position_correlation_probability, 0.1 / its_roughness);

                        if(rRec.nextSample1D() < position_correlation_probability){
                            use_positional_correlation = true;
                        } else {
                            use_positional_correlation = false;
                        }
                    } else {
                        if(its_roughness > 0.5){
                            use_positional_correlation = true;
                        }
                    }
                    // Vector3 reflected_dir = reflect(-path1.ray.d, path1.its.shFrame.n);
                    // const RayDifferential ray_mirror = RayDifferential(path1.its.p, reflected_dir, path1.ray.time);
                    // Intersection its_mirror;
                    // scene->rayIntersect(ray_mirror, its_mirror);

                    // Float reflected_cos_o = 1.0f;
                    // Float reflected_dist = -1.0f;
                    // if(scene->rayIntersect(ray_mirror, its_mirror)){
                    //     reflected_dist = its_mirror.t;
                    //     reflected_cos_o = std::abs(dot(its_mirror.shFrame.n, ray_mirror.d));
                    // } else {
                    //     reflected_dist = 10.0f;
                    //     reflected_cos_o = 1.0f;
                    // }

                    // if(its_mirror.isValid()){
                    //     Intersection its_mirror2p = its_mirror;
                    //     its_mirror2p.adjustTime(path2.ray.time);
                        
                    //     const BSDF *bsdf1 = path1.its.getBSDF(path1.ray);
                    //     const BSDF *bsdf2 = path2.its.getBSDF(path2.ray);

                    //     Vector3 reflected_dir2 = reflect(-path2.ray.d, path2.its.shFrame.n);
                    //     const RayDifferential ray_mirror2d = RayDifferential(path2.its.p, reflected_dir2, path2.ray.time);
                    //     const RayDifferential ray_mirror2p = RayDifferential(path2.its.p, normalize(its_mirror2p.p - path2.its.p), path2.ray.time);

                    //     Intersection its_mirror2d;
                    //     scene->rayIntersect(ray_mirror2d, its_mirror2d);

                    //     BSDFSamplingRecord bRec1(path1.its, path1.its.toLocal(ray_mirror.d), ERadiance);
                    //     BSDFSamplingRecord bRec1m(its_mirror, its_mirror.toLocal(normalize(light_origin - its_mirror.p)), ERadiance);
                    //     Spectrum bsdfVal1m = bsdf1->eval(bRec1);
                    //     const BSDF *bsdf1m = its_mirror.getBSDF(ray_mirror);
                    //     Spectrum bsdfVal1l = bsdf1m->eval(bRec1m);

                    //     BSDFSamplingRecord bRec2p(path2.its, path2.its.toLocal(ray_mirror2p.d), ERadiance);
                    //     BSDFSamplingRecord bRec2mp(its_mirror2p, its_mirror2p.toLocal(normalize(light_origin - its_mirror2p.p)), ERadiance);
                    //     Spectrum bsdfVal2mp = bsdf2->eval(bRec2p);
                    //     const BSDF *bsdf2mp = its_mirror2p.getBSDF(ray_mirror2p);
                    //     Spectrum bsdfVal2lp = bsdf2mp->eval(bRec2mp);
                        
                    //     Float p1 = 0;
                    //     Float p2p = 0;
                    //     Float p2d = 0;
                    //     if(its_mirror2d.isValid()){
                    //         BSDFSamplingRecord bRec2d(path2.its, path2.its.toLocal(ray_mirror2d.d), ERadiance);
                    //         BSDFSamplingRecord bRec2md(its_mirror2d, its_mirror2d.toLocal(normalize(light_origin - its_mirror2d.p)), ERadiance);
                    //         Spectrum bsdfVal2md = bsdf2->eval(bRec2d);
                    //         const BSDF *bsdf2md = its_mirror2d.getBSDF(ray_mirror2d);
                    //         Spectrum bsdfVal2ld = bsdf2md->eval(bRec2md);
                    //         p2d = (bsdfVal2md * bsdfVal2ld)[0];
                    //     }
                    //     p1 = (bsdfVal1m * bsdfVal1l)[0];
                    //     p2p = (bsdfVal2mp * bsdfVal2lp)[0];

                    //     if(std::abs(p1 - p2p) < std::abs(p1 - p2d)){
                    //         use_positional_correlation = true;
                    //     } else {
                    //         use_positional_correlation = true;
                    //     }
                    // }
                    // Float area = its_roughness * reflected_dist * reflected_dist / (reflected_cos_o);

                    // Float position_correlation_probability = 6.0f * area;
                    
                }
            }
           

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
            else if(path1.path_terminated && !path2.path_terminated){
                break;
            }
            else if(!path1.path_terminated && path2.path_terminated){
                path1.set_path_pdf_as(path2, 0.0);
                path2.set_path_pdf_as(path2, 0.0);
                path2.path_throughput *= 0;
            }

            // direct emission
            if(rRec.depth == 0 && !m_hideEmitters){
                Li += accumulate_next_ray_Li(path1, path2);
                Li += accumulate_next_ray_Li(path2, path1);
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */
            Point2 nee_sample = rRec.nextSample2D();
            path1.nee_trace(nee_sample);

            // For nee, positional correlation is equal to sampler correlation.
            if(use_positional_correlation){
                path2.nee_trace(nee_sample);
                PathTracePart::calculate_nee_pdf_positional(path1, path2);
            } else {
                path2.nee_trace(nee_sample);
                PathTracePart::calculate_nee_pdf_sampler(path1, path2);
            }
            
            // Accumulate nee Li
            Li += accumulate_nee_Li(path1, path2);
            Li += accumulate_nee_Li(path2, path1);

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            Point2 bsdf_sample = rRec.nextSample2D();
            path1.get_next_ray_from_sample(bsdf_sample);

            if(path1.path_terminated){
                break;
            }

            if(use_positional_correlation){
                path2.get_next_ray_from_its(path1.its);
            } else {
                path2.get_next_ray_from_sample(bsdf_sample);
            }

            if(path2.path_terminated){
                path2.path_throughput *= 0.0;
                path2.set_path_pdf_as(path2, 0.0);
                path1.set_path_pdf_as(path2, 0.0);
            }

            if(use_positional_correlation){
                PathTracePart::calculate_next_ray_pdf_positional(path1, path2);
                path1.set_path_pdf_as(path1, path1.bsdfPdf);
                PathTracePart::update_path_pdf_positional(path1, path2);
            } else {
                PathTracePart::calculate_next_ray_pdf_sampler(path1, path2);
                path1.set_path_pdf_as(path1, path1.bsdfPdf);
                PathTracePart::update_path_pdf_sampler(path1, path2);
            }

            // Accumulate next ray Li
            Li += accumulate_next_ray_Li(path1, path2);
            Li += accumulate_next_ray_Li(path2, path1);

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec2.type = RadianceQueryRecord::ERadianceNoEmission;
            
            rRec.depth++;
            rRec2.depth++;
        }
        return Li;
    }

    // Trace ray with two paired samples + two MIS pairs (total 5)
    Spectrum Li_with_MIS(
            const RayDifferential &r1,
            const RayDifferential &r2, 
            RadianceQueryRecord &rRec
        ) const {

        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Spectrum Li(0.0f);
        RadianceQueryRecord rRec2 = rRec;
        RadianceQueryRecord rRec3 = rRec;
        RadianceQueryRecord rRec4 = rRec;
        RadianceQueryRecord rRec5 = rRec;

        // path 1 (primal)
        RayDifferential ray1(r1);
        rRec.rayIntersect(ray1);
        PathTracePart path1(ray1, rRec, 0);

        // path 2 (positional correlation)
        RayDifferential ray2(r2);
        rRec2.rayIntersect(ray2);
        PathTracePart path2(ray2, rRec2, 1);

        // path 3 (sampler correlation)
        RayDifferential ray3(r2);
        rRec3.rayIntersect(ray3);
        PathTracePart path3(ray3, rRec3, 2);
        
        // path 4 (primal that has positional correlation as sampler correlation)
        // RayDifferential ray4(r1);
        // rRec4.rayIntersect(ray4);
        // PathTracePart path4(ray4, rRec4, 3);

        // path 5 (primal that has sampler correlation as positional correlation)
        RayDifferential ray5(r1);
        rRec5.rayIntersect(ray5);
        PathTracePart path5(ray5, rRec5, 4);

        while(rRec.depth <= m_maxDepth || m_maxDepth < 0){
            if(path1.path_terminated){
                break;
            }

            if(path2.path_terminated){
                path1.set_path_pdf_as(path2, 0.0);
                path2.set_path_pdf_as(path2, 0.0);
                path3.set_path_pdf_as(path2, 0.0);
                path2.path_throughput *= 0;
            }

            if(path3.path_terminated){
                path1.set_path_pdf_as(path3, 0.0);
                path2.set_path_pdf_as(path3, 0.0);
                path3.set_path_pdf_as(path3, 0.0);
                path3.path_throughput *= 0;
            }

            // if(path4.path_terminated){
            //     path3.set_path_pdf_as(path2, 0.0);
            //     path4.set_path_pdf_as(path4, 0.0);
            // }

            if(path5.path_terminated){
                path2.set_path_pdf_as(path3, 0.0);
                path5.set_path_pdf_as(path5, 0.0);
            }

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(path1.ray.d, path1.its.geoFrame.n)
                    * Frame::cosTheta(path1.its.wi) >= 0)) {
                break;
            }

            // direct emission
            if(rRec.depth == 0 && !m_hideEmitters){
                Li += 0.5 * accumulate_next_ray_Li(path1, path2);
                Li += 0.5 * accumulate_next_ray_Li(path1, path3);
                Li += 0.5 * accumulate_next_ray_Li(path2, path1);
                Li += 0.5 * accumulate_next_ray_Li(path3, path1);
            }


            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */
            Point2 nee_sample = rRec.nextSample2D();
            path1.nee_trace(nee_sample);

            path2.nee_trace(nee_sample);
            PathTracePart::calculate_nee_pdf_positional(path1, path2);

            path3.nee_trace(nee_sample);
            PathTracePart::calculate_nee_pdf_sampler(path1, path3);
            
            // Accumulate nee Li
            Float mis23 = miWeight(path2.path_pdf_as(path2), path2.path_pdf_as(path3), 2);
            Li += 0.5 * accumulate_nee_Li(path1, path2);
            Li += mis23 * accumulate_nee_Li(path2, path1);
            
            Float mis32 = miWeight(path3.path_pdf_as(path3), path3.path_pdf_as(path2), 2);
            Li += 0.5 * accumulate_nee_Li(path1, path3);
            Li += mis32 * accumulate_nee_Li(path3, path1);


            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            Point2 bsdf_sample = rRec.nextSample2D();
            path1.get_next_ray_from_sample(bsdf_sample);

            if(path1.path_terminated){
                break;
            }
            if(!path2.path_terminated){
                path2.get_next_ray_from_its(path1.its);
            }
            if(!path3.path_terminated){
                path3.get_next_ray_from_sample(bsdf_sample);
            }
           
            if(!path2.path_terminated){
                // path4.get_next_ray_from_sample(path2.sample);
            } else {
                path2.path_throughput *= 0.0;
                path1.set_path_pdf_as(path2, 0.0);
                path2.set_path_pdf_as(path2, 0.0);
                path3.set_path_pdf_as(path2, 0.0);
            }
            if(!path3.path_terminated){
                path5.get_next_ray_from_its(path3.its);
            } else {
                path3.path_throughput *= 0.0;
                path1.set_path_pdf_as(path3, 0.0);
                path2.set_path_pdf_as(path3, 0.0);
                path3.set_path_pdf_as(path3, 0.0);
            }
            PathTracePart::calculate_next_ray_pdf_positional(path1, path2);
            PathTracePart::calculate_next_ray_pdf_sampler(path1, path3);
            path1.set_path_pdf_as(path1, path1.bsdfPdf);
            PathTracePart::update_path_pdf_positional(path1, path2);
            PathTracePart::update_path_pdf_sampler(path1, path3);
            path2.set_path_pdf_as(path3, path2.bsdfPdf);
            path3.set_path_pdf_as(path2, path5.bsdfPdf * path5.G / path3.G);

            //PathTracePart::update_path_pdf(path1, path2, 1);
            //PathTracePart::update_path_pdf(path1, path3, path1.G / path3.G);

            mis23 = miWeight(path2.path_pdf_as(path2), path2.path_pdf_as(path3), 2);
            mis32 = miWeight(path3.path_pdf_as(path3), path3.path_pdf_as(path2), 2);
            
            // Accumulate next ray Li
            Li += 0.5 * accumulate_next_ray_Li(path1, path2);
            Li += mis23 * accumulate_next_ray_Li(path2, path1);
            
            Li += 0.5 * accumulate_next_ray_Li(path1, path3);
            Li += mis32 * accumulate_next_ray_Li(path3, path1);



            rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec2.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec3.type = RadianceQueryRecord::ERadianceNoEmission;
            
            rRec.depth++;
            rRec2.depth++;
            rRec3.depth++;
            
            // PathTracePart::calculate_bsdf_pdf_as_other(path4, path2, path3, path4.G / path2.G);
            // PathTracePart::calculate_bsdf_pdf_as_other(path5, path3, path2, 1);
            // PathTracePart::calculate_bsdf_pdf_as_other(path5, path3, 1);
            //path2.set_path_pdf_as(path3, );
        }
        return Li;
    }

    // Trace ray with sampler correlated fashion.
    Spectrum Li_with_path_sampler_correlation(const RayDifferential &r, RadianceQueryRecord &_rRec) const
    {
        RadianceQueryRecord rRec = _rRec;
        int n_antithetic = this->m_antithetic_shifts.size();
        rRec.sampler->saveState();

        // primal path
        RayDifferential ray(r);
        Spectrum Li = Li_with_single_sample(ray, rRec);

        // antithetic paths
        for(float antithetic_shift : this->m_antithetic_shifts){
            RadianceQueryRecord rRec = _rRec;
            rRec.sampler->loadSavedState();
            RayDifferential sensorRay;
            rRec.scene->getSensor()->sampleRayDifferential(
                sensorRay, _rRec.samplePos, _rRec.apertureSample, std::fmod(_rRec.timeSample + antithetic_shift, 1.0));
            sensorRay.scaleDifferential(_rRec.diffScaleFactor);

            Li += Li_with_single_sample(sensorRay, rRec);
        }
        for(float antithetic_shift : this->m_antithetic_shifts){
            rRec.sampler->advance();
        }
        return Li * (1.0) / (n_antithetic + 1);
    }


    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        // 1. Time sampling
        Float t0 = rRec.timeSample;
        std::vector<Float> sampled_times;
        int n_time_samples = this->m_antithetic_shifts.size() + 1;

        // (1) uniform sampling
        if(strcmp(m_time_sampling_mode.c_str(), "uniform") == 0){
            for(int i=0; i<n_time_samples; i++){
                sampled_times.push_back(rRec.nextSample1D());
            }
        }

        // (2) stratified sampling
        else if(strcmp(m_time_sampling_mode.c_str(), "stratified") == 0){
            for(int i=0; i<n_time_samples; i++){
                sampled_times.push_back((rRec.nextSample1D() + i) * 1.0 / n_time_samples);
            }
        }
        // (3) antithetic sampling
        else if(strcmp(m_time_sampling_mode.c_str(), "antithetic") == 0){
            sampled_times.push_back(t0);
            for(int i=0; i<n_time_samples - 1; i++){
                sampled_times.push_back(t0 + m_antithetic_shifts.at(i));
            }
        }
        // (4) antithetic sampling mirror
        else if(strcmp(m_time_sampling_mode.c_str(), "antithetic_mirror") == 0){
            sampled_times.push_back(t0);
            sampled_times.push_back(1.0 - t0 + m_antithetic_shifts.at(0));
        }

        // 2. Spatial correlation

        // (1) no correlation
        if(strcmp(m_spatial_correlation_mode.c_str(), "none") == 0){
            Spectrum Li(0.0f);
            for(int i=0; i<n_time_samples; i++){
                RayDifferential r2;
                RadianceQueryRecord rRec2 = rRec;

                Point2 samplePos = Point2(rRec.offset) + Vector2(rRec2.nextSample2D());
                Point2 apertureSample = rRec2.nextSample2D();

                rRec2.scene->getSensor()->sampleRayDifferential(
                    r2, samplePos, apertureSample, std::fmod(sampled_times.at(i), 1.0));
                r2.scaleDifferential(rRec.diffScaleFactor);
                Li += Li_with_single_sample(r2, rRec2);
            }
            return Li * (1.0) / n_time_samples;
        }

        // (2) pixel correlation
        else if(strcmp(m_spatial_correlation_mode.c_str(), "pixel") == 0){
            Spectrum Li(0.0f);
            for(int i=0; i<n_time_samples; i++){
                RayDifferential r2;
                RadianceQueryRecord rRec2 = rRec;

                Point2 samplePos = rRec.samplePos;
                Point2 apertureSample = rRec.apertureSample;

                rRec2.scene->getSensor()->sampleRayDifferential(
                    r2, samplePos, apertureSample, std::fmod(sampled_times.at(i), 1.0));
                r2.scaleDifferential(rRec.diffScaleFactor);
                Li += Li_with_single_sample(r2, rRec2);
            }
            return Li * (1.0) / n_time_samples;
        }

        // (3) sampler correlation
        else if(strcmp(m_spatial_correlation_mode.c_str(), "sampler") == 0){
            Spectrum Li(0.0f);
            rRec.sampler->saveState();

            for(int i=0; i<n_time_samples; i++){
                RayDifferential r2;
                RadianceQueryRecord rRec2 = rRec;
                rRec.sampler->loadSavedState();

                Point2 samplePos = rRec.samplePos;
                Point2 apertureSample = rRec.apertureSample;

                rRec2.scene->getSensor()->sampleRayDifferential(
                    r2, samplePos, apertureSample, std::fmod(sampled_times.at(i), 1.0));
                r2.scaleDifferential(rRec.diffScaleFactor);
                Li += Li_with_single_sample(r2, rRec2);
            }
            for(int i=0; i<n_time_samples-1; i++){
                rRec.sampler->advance();
            }
            return Li * (1.0) / n_time_samples;
        }

        // (4) per-ray correlation (sampler / position / mis / stochastic)
        else if(strcmp(m_spatial_correlation_mode.c_str(), "mis") == 0){
            RayDifferential r1;
            rRec.scene->getSensor()->sampleRayDifferential(
                r1, rRec.samplePos, rRec.apertureSample, std::fmod(sampled_times.at(0), 1.0));
            r1.scaleDifferential(rRec.diffScaleFactor);

            RayDifferential r2;
            rRec.scene->getSensor()->sampleRayDifferential(
                r2, rRec.samplePos, rRec.apertureSample, std::fmod(sampled_times.at(1), 1.0));
            r2.scaleDifferential(rRec.diffScaleFactor);
            return Li_with_MIS(r1, r2, rRec);
        } 
        else {
            RayDifferential r1;
            rRec.scene->getSensor()->sampleRayDifferential(
                r1, rRec.samplePos, rRec.apertureSample, std::fmod(sampled_times.at(0), 1.0));
            r1.scaleDifferential(rRec.diffScaleFactor);

            RayDifferential r2;
            rRec.scene->getSensor()->sampleRayDifferential(
                r2, rRec.samplePos, rRec.apertureSample, std::fmod(sampled_times.at(1), 1.0));
            r2.scaleDifferential(rRec.diffScaleFactor);
            return Li_with_paired_sample(r1, r2, rRec);
        }
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "ToFAntitheticPathTracer[" << endl
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
    Float m_time;
    Float m_antithetic_shift;
    std::vector<float> m_antithetic_shifts;
    
    bool m_low_frequency_component_only;
    bool m_force_constant_attenuation;
    bool m_correlate_time_samples;
    bool m_antithetic_sampling_by_shift;
    bool m_preserve_primal_mis_weight_to_one;
    float m_primal_antithetic_mis_power;
    float m_hetero_frequency;

    std::string m_mis_method;
    std::string m_time_sampling_mode;
    std::string m_spatial_correlation_mode;

    int m_wave_function_type;

    // std::string m_antithetic_sampling_mode;
};

MTS_IMPLEMENT_CLASS_S(ToFAntitheticPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ToFAntitheticPathTracer, "ToF antithetic path tracer");
MTS_NAMESPACE_END


