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

static StatsCounter avgPathLength("DifferencePath tracer", "Average path length", EAverage);

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


class DifferencePathTracer : public MonteCarloIntegrator {
public:
    DifferencePathTracer(const Properties &props)
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

        m_difference_target = props.getString("differenceTarget", "time");
    }

    /// Unserialize from a binary data stream
    DifferencePathTracer(Stream *stream, InstanceManager *manager)
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
        return 1;
        //return (ray_time > 0.5 * m_time) ? 1.0 : -1.0;
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


        PathTracePart path1_next(ray1, rRec, 0);

        Point3 light_origin = ray1.o;
        while(rRec.depth <= m_maxDepth || m_maxDepth < 0){

            Point2 bsdf_sample = rRec.nextSample2D();
            path1_next = path1;
            path1_next.get_next_ray_from_sample(bsdf_sample);

            bool use_positional_correlation = false;
            if(strcmp(m_spatial_correlation_mode.c_str(), "position") == 0){
                use_positional_correlation = true;
            }
            if(path1.its.isValid() && path1_next.its.isValid()){
                const BSDF *bsdf = path1.its.getBSDF(path1.ray);
                Float its_roughness = bsdf->getRoughness(path1.its, 0);
                const BSDF *bsdf_next = path1_next.its.getBSDF(path1_next.ray);
                Float its_roughness_next = bsdf_next->getRoughness(path1_next.its, 0);
                // if(its_roughness == 0.0){
                //     use_positional_correlation = false;
                // }

                its_roughness = std::min(its_roughness, 2.0);

                if(strcmp(m_spatial_correlation_mode.c_str(), "selective") == 0){
                    if(its_roughness > 0.5 && its_roughness_next > 0.5){
                        use_positional_correlation = true;
                    }
                }

                if(strcmp(m_spatial_correlation_mode.c_str(), "adaptive") == 0){
                    if(rRec.depth == 1){
                        if(rRec.use_positional_correlation_probability > 0.5){
                            use_positional_correlation = true;
                        } else {
                            use_positional_correlation = false;
                        }
                    } else {
                        if(its_roughness > 0.5){
                            use_positional_correlation = true;
                        }
                    }
                }

                if(strcmp(m_spatial_correlation_mode.c_str(), "mixed") == 0){
                    Intersection transformed_its = path1_next.its;
                    transformed_its.adjustTime(path2.ray.time);
                    
                    Vector3 mv1 = transformed_its.p - path1_next.its.p;
                    
                    Vector3 mv2 = mv1 - dot(mv1, path1_next.ray.d) * path1_next.ray.d;
                    Vector3 mv3 = mv2 - dot(mv2, path1_next.its.geoFrame.n) * path1_next.its.geoFrame.n;

                    Vector3 mv_local = path1_next.its.toLocal(mv3);
                    Vector2 mv = Vector2(mv_local.x, mv_local.y);
                    Vector2 uv = path1_next.its.uv;
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
                Li -= accumulate_next_ray_Li(path2, path1);
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
            Li -= accumulate_nee_Li(path2, path1);

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            path1 = path1_next;

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
            Li -= accumulate_next_ray_Li(path2, path1);

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec2.type = RadianceQueryRecord::ERadianceNoEmission;
            
            rRec.depth++;
            rRec2.depth++;
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
        Float t0 = rRec.nextSample1D();//rRec.timeSample;
        std::vector<Float> sampled_signs;
        std::vector<Float> sampled_times;
        std::vector<Vector2> sampled_offsets;
        
        Float reversed = 1.0;

        int n_samples = 2;
        

        if (strcmp(m_difference_target.c_str(), "time") == 0){
            if(t0 > 0.5){
                sampled_times.push_back(1.0);
                sampled_times.push_back(0.0);
                reversed=1.0;
            } else {
                sampled_times.push_back(0.0);
                sampled_times.push_back(1.0);
                reversed=-1.0;
            }    
            sampled_offsets.push_back(Vector2(0, 0));
            sampled_offsets.push_back(Vector2(0, 0));
        }
        else if (strcmp(m_difference_target.c_str(), "x") == 0){
            if(t0 > 0.5){
                sampled_offsets.push_back(Vector2(0, 0));
                sampled_offsets.push_back(Vector2(1, 0));
                reversed=1.0;
            } else {
                sampled_offsets.push_back(Vector2(1, 0));
                sampled_offsets.push_back(Vector2(0, 0));
                reversed=-1.0;
            }    
            sampled_times.push_back(0);
            sampled_times.push_back(0);
        }
        else if (strcmp(m_difference_target.c_str(), "y") == 0){
            if(t0 > 0.5){
                sampled_offsets.push_back(Vector2(0, 0));
                sampled_offsets.push_back(Vector2(0, 1));
                reversed=1.0;
            } else {
                sampled_offsets.push_back(Vector2(0, 1));
                sampled_offsets.push_back(Vector2(0, 0));
                reversed=-1.0;
            }    
            sampled_times.push_back(0);
            sampled_times.push_back(0);
        }

        // 2. Spatial correlation

        // (1) no correlation
        if(strcmp(m_spatial_correlation_mode.c_str(), "none") == 0){
            Spectrum Li(0.0f);
            for(int i=0; i<n_samples; i++){
                RayDifferential r2;
                RadianceQueryRecord rRec2 = rRec;

                Point2 samplePos = Point2(rRec.offset) + Vector2(rRec2.nextSample2D()) + sampled_offsets.at(i);
                Point2 apertureSample = rRec2.nextSample2D();

                rRec2.scene->getSensor()->sampleRayDifferential(
                    r2, samplePos, apertureSample, sampled_times.at(i));
                r2.scaleDifferential(rRec.diffScaleFactor);
                Li += Li_with_single_sample(r2, rRec2) * (i==0?1.0:-1.0);
            }
            return Li * (1.0) / n_samples * reversed;
        }

        // (3) sampler correlation
        else if(strcmp(m_spatial_correlation_mode.c_str(), "sampler") == 0){
            Spectrum Li(0.0f);
            rRec.sampler->saveState();

            for(int i=0; i<n_samples; i++){
                RayDifferential r2;
                RadianceQueryRecord rRec2 = rRec;
                rRec.sampler->loadSavedState();

                Point2 samplePos = rRec.samplePos + sampled_offsets.at(i);
                Point2 apertureSample = rRec.apertureSample;

                rRec2.scene->getSensor()->sampleRayDifferential(
                    r2, samplePos, apertureSample, sampled_times.at(i));
                r2.scaleDifferential(rRec.diffScaleFactor);
                Li += Li_with_single_sample(r2, rRec2) * (i==0?1.0:-1.0);
            }
            for(int i=0; i<n_samples-1; i++){
                rRec.sampler->advance();
            }
            return Li * (1.0) / n_samples * reversed;
        }

        // (4) per-ray correlation (sampler / position / mis / stochastic)
        else {
            RayDifferential r1;
            rRec.scene->getSensor()->sampleRayDifferential(
                r1, rRec.samplePos + sampled_offsets.at(0), rRec.apertureSample, sampled_times.at(0));
            r1.scaleDifferential(rRec.diffScaleFactor);

            RayDifferential r2;
            rRec.scene->getSensor()->sampleRayDifferential(
                r2, rRec.samplePos + sampled_offsets.at(1), rRec.apertureSample, sampled_times.at(1));
            r2.scaleDifferential(rRec.diffScaleFactor);

            return Li_with_paired_sample(r1, r2, rRec) * reversed;
        }
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "DifferencePathTracer[" << endl
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
    std::string m_difference_target;

    int m_wave_function_type;

    // std::string m_antithetic_sampling_mode;
};

MTS_IMPLEMENT_CLASS_S(DifferencePathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(DifferencePathTracer, "Difference path tracer");
MTS_NAMESPACE_END


