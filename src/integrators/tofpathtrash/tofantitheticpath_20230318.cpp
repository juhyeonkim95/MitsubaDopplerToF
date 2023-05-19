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
        m_needOffset = true;
        m_time = props.getFloat("time");
        m_illumination_modulation_frequency_mhz = props.getFloat("w_g", 30.0f);
        m_illumination_modulation_scale = props.getFloat("g_1", 0.5f);
        m_illumination_modulation_offset = props.getFloat("g_0", 0.5f);

        m_sensor_modulation_frequency_mhz = props.getFloat("w_f", 30.0f);
        m_sensor_modulation_scale = props.getFloat("f_1", 0.5f);
        m_sensor_modulation_offset = props.getFloat("f_0", 0.5f);
        m_sensor_modulation_phase_offset = props.getFloat("f_phase_offset", 0.0f);
        m_antithetic_shift = props.getFloat("antitheticShift", 0.5f);
        std::string antithetic_shift_string = props.getString("antitheticShifts", "0.5");
        m_antithetic_shifts = parse_float_array_from_string(antithetic_shift_string);

        m_low_frequency_component_only = props.getBoolean("low_frequency_component_only", false);
        m_is_objects_transformed_for_time = props.getBoolean("is_object_transformed_for_time", false);
        m_force_constant_attenuation = props.getBoolean("force_constant_attenuation", false);
        m_antithetic_sampling_by_shift = props.getBoolean("antitheticSamplingByShift", true);
        m_preserve_primal_mis_weight_to_one = props.getBoolean("preservePrimalMISWeightToOne", true);
        m_mis_method = props.getString("misMethod", "mix_all");

        m_antithetic_sampling_mode = props.getString("antitheticSamplingMode", "position");

        m_primal_antithetic_mis_power = props.getFloat("primalAntitheticMISPower", 1.0f);
    }

    /// Unserialize from a binary data stream
    ToFAntitheticPathTracer(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    Float evalModulationWeight(Float &ray_time, Float &path_length) const{
        Float w_g = 2 * M_PI * m_illumination_modulation_frequency_mhz * 1e6;
        Float w_f = 2 * M_PI * m_sensor_modulation_frequency_mhz * 1e6;
        Float w_d = 2 * M_PI * (m_sensor_modulation_frequency_mhz - m_illumination_modulation_frequency_mhz) * 1e6;
        Float phi = (2 * M_PI * m_illumination_modulation_frequency_mhz) / 300 * path_length;

        if(m_low_frequency_component_only){
            Float fg_t = 0.25 * std::cos(w_d * ray_time + m_sensor_modulation_phase_offset + phi);
            return fg_t;
        } 
        
        Float g_t = 0.5 * std::cos(w_g * ray_time - phi) + 0.5;
        Float f_t = std::cos(w_f * ray_time + m_sensor_modulation_phase_offset);
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


    Spectrum Li_single(const RayDifferential &r, RadianceQueryRecord &rRec) const {
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


    Spectrum Li_helper(const RayDifferential &r, RadianceQueryRecord &rRec, Float ray2_timeSample) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Spectrum Li(0.0f);
        RadianceQueryRecord rRec2 = rRec;

        // path 1
        RayDifferential ray1(r);
        rRec.rayIntersect(ray1);
        PathTracePart path1(ray1, rRec, 0);

        // path 2
        RayDifferential ray2;
        rRec.scene->getSensor()->sampleRayDifferential(
            ray2, rRec.samplePos, rRec.apertureSample, std::fmod(rRec.timeSample + ray2_timeSample, 1.0));
        ray2.scaleDifferential(rRec.diffScaleFactor);
        rRec2.rayIntersect(ray2);
        PathTracePart path2(ray2, rRec2, 1);
        bool use_positional_correlation = false;
        //(strcmp(m_antithetic_sampling_mode.c_str(), "position") == 0);
        
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
            else if(path1.path_terminated && !path2.path_terminated){
                break;
                // path1.set_path_pdf_as(path1, 0.0);
                // path2.set_path_pdf_as(path1, 0.0);
                // path1.path_throughput *= 0;
                // PathTracePart path3 = path2;
                // path2 = path1;
                // path1 = path3;
                // std::swap(path1, path2);
            }
            else if(!path1.path_terminated && path2.path_terminated){
                // break;
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

            Float jacobian;

            // For nee, positional correlation is equal to sampler correlation.
            if(use_positional_correlation){
                path2.nee_trace(nee_sample);
                jacobian = path2.G_nee / path1.G_nee;
            } else {
                path2.nee_trace(nee_sample);
                jacobian = 1;//path1.G_nee / path2.G_nee;
            }
            PathTracePart::calculate_nee_pdf(path1, path2, jacobian);

            // Accumulate nee Li
            Li += accumulate_nee_Li(path1, path2);
            Li += accumulate_nee_Li(path2, path1);

            if(rRec.depth + 1>=m_maxDepth){
                break;
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            use_positional_correlation = false;
            if(strcmp(m_antithetic_sampling_mode.c_str(), "position") == 0){
                use_positional_correlation = true;
            }
            if(!check_consistency(path1.its, path2.its) || (path1.its.getBSDF(path1.ray) != path2.its.getBSDF(path2.ray)) ){
                // use_positional_correlation = false;
            }

            Point2 bsdf_sample = rRec.nextSample2D();
            path1.get_next_ray_from_sample(bsdf_sample);

            const BSDF *bsdf = path1.prev_its.getBSDF(path1.ray);
            Float its_roughness = bsdf->getRoughness(path1.prev_its, 0);
            
            if ((strcmp(m_antithetic_sampling_mode.c_str(), "adaptive") == 0)){
                use_positional_correlation = (its_roughness > 0.1);
            }

            // if(path1.its.t < 0.1){
            //     use_positional_correlation = false;
            // }

            // if(path1.prev_its.shFrame.n.y > 0.9){
            //     printPoint(path1.its.p, path2.its.p);
            //     use_positional_correlation = false;
            // }
        
            if(path1.path_terminated){
                break;
                use_positional_correlation = false;
                path1.path_throughput *= 0.0;
                path1.set_path_pdf_as(path1, 0.0);
                path2.set_path_pdf_as(path1, 0.0);
            }
            // if(its_roughness == 0.0){
            //     use_positional_correlation = false;
            // }

            if(use_positional_correlation){
                path2.get_next_ray_from_its(path1.its);
                jacobian = path2.G / path1.G;//1;
            } else {
                path2.get_next_ray_from_sample(bsdf_sample);
                jacobian = 1;//path1.G / path2.G;
            }

            if(path2.path_terminated){
                path2.path_throughput *= 0.0;
                path2.set_path_pdf_as(path2, 0.0);
                path1.set_path_pdf_as(path2, 0.0);
                // return Spectrum(100.0);
            }

            PathTracePart::calculate_next_ray_pdf(path1, path2, jacobian);
            
            path1.set_path_pdf_as(path1, path1.bsdfPdf);
            PathTracePart::update_path_pdf(path1, path2, jacobian);

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

    Spectrum Li_helper_mis_new(const RayDifferential &r, RadianceQueryRecord &rRec, Float ray2_timeSample) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Spectrum Li(0.0f);
        RadianceQueryRecord rRec2 = rRec;
        RadianceQueryRecord rRec3 = rRec;
        RadianceQueryRecord rRec4 = rRec;
        RadianceQueryRecord rRec5 = rRec;

        // path 1 (primal)
        RayDifferential ray1(r);
        rRec.rayIntersect(ray1);
        PathTracePart path1(ray1, rRec, 0);

        // path 2 (positional correlation)
        RayDifferential ray2;
        rRec.scene->getSensor()->sampleRayDifferential(
            ray2, rRec.samplePos, rRec.apertureSample, std::fmod(rRec.timeSample + ray2_timeSample, 1.0));
        ray2.scaleDifferential(rRec.diffScaleFactor);
        rRec2.rayIntersect(ray2);
        PathTracePart path2(ray2, rRec2, 1);

        // path 3 (sampler correlation)
        RayDifferential ray3;
        rRec.scene->getSensor()->sampleRayDifferential(
            ray3, rRec.samplePos, rRec.apertureSample, std::fmod(rRec.timeSample + ray2_timeSample, 1.0));
        ray3.scaleDifferential(rRec.diffScaleFactor);
        rRec3.rayIntersect(ray3);
        PathTracePart path3(ray3, rRec3, 2);
        
        // path 4 (primal that has positional correlation as sampler correlation)
        RayDifferential ray4(r);
        rRec4.rayIntersect(ray4);
        PathTracePart path4(ray4, rRec4, 3);

        // path 5 (primal that has sampler correlation as positional correlation)
        RayDifferential ray5(r);
        rRec5.rayIntersect(ray5);
        PathTracePart path5(ray5, rRec5, 4);

        while(rRec.depth <= m_maxDepth || m_maxDepth < 0){
            if(!path1.its.isValid() || path1.path_terminated){
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
            //     path4.set_path_pdf_as(path4, 0.0);
            // }

            // if(path5.path_terminated){
            //     path5.set_path_pdf_as(path5, 0.0);
            // }

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
            PathTracePart::calculate_nee_pdf(path1, path2, 1);

            path3.nee_trace(nee_sample);
            PathTracePart::calculate_nee_pdf(path1, path3, path1.G_nee / path3.G_nee);
            
            // Accumulate nee Li
            Float mis23 = miWeight(path2.path_pdf_as(path2), path2.path_pdf_as(path3));
            Li += 0.5 * accumulate_nee_Li(path1, path2);
            Li += mis23 * accumulate_nee_Li(path2, path1);
            
            Float mis32 = miWeight(path3.path_pdf_as(path3), path3.path_pdf_as(path2));
            Li += 0.5 * accumulate_nee_Li(path1, path3);
            Li += mis32 * accumulate_nee_Li(path3, path1);

            if(rRec.depth + 1>=m_maxDepth){
                break;
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            Point2 bsdf_sample = rRec.nextSample2D();
            path1.get_next_ray_from_sample(bsdf_sample);

            if(path1.path_terminated){
                break;
            }
            
            path2.get_next_ray_from_its(path1.its);
            PathTracePart::calculate_next_ray_pdf(path1, path2, 1);
            
            path3.get_next_ray_from_sample(bsdf_sample);
            PathTracePart::calculate_next_ray_pdf(path1, path3, path1.G / path3.G);

            if(!path2.path_terminated){
                path4.get_next_ray_from_sample(path2.sample);
            }
            if(!path3.path_terminated){
                path5.get_next_ray_from_its(path3.its);
            }

            path1.set_path_pdf_as(path1, path1.bsdfPdf * path1.G);
            PathTracePart::update_path_pdf(path1, path2, 1);
            PathTracePart::update_path_pdf(path1, path3, path1.G / path3.G);

            // Accumulate next ray Li
            Li += 0.5 * accumulate_next_ray_Li(path1, path2);
            Li += mis23 * accumulate_next_ray_Li(path2, path1);
            
            Li += 0.5 * accumulate_next_ray_Li(path1, path3);
            Li += mis32 * accumulate_next_ray_Li(path3, path1);

            path2.set_path_pdf_as(path3, path4.bsdfPdf * path2.G);
            path3.set_path_pdf_as(path2, path5.bsdfPdf * path5.G);

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

    Spectrum Li_helper_mis(const RayDifferential &r, RadianceQueryRecord &rRec, Float ray2_time) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Spectrum Li(0.0f);
        RadianceQueryRecord rRec2 = rRec;
        RadianceQueryRecord rRec3 = rRec;

        // path 1
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;
        Float path_length = 0;
        path_length += its.t;
        Float path_pdf = 1.0f;
        Spectrum path_throughput(1.0f);
        Float G = 1.0f;
        
        // path 2 (antithetic path)
        bool path2_blocked = false;
        Intersection &its2 = rRec2.its;
        RayDifferential ray2(r);
        ray2.time = ray2_time;
        rRec2.rayIntersect(ray2);
        ray2.mint = Epsilon;
        Float path_length2 = 0;
        path_length2 += its2.t;
        Float path_pdf2 = 1.0f;
        Spectrum path_throughput2(1.0f);
        Float G2 = 1.0f;

        // path 3 (antithetic, direction correlation)
        bool path3_blocked = false;
        Intersection &its3 = rRec3.its;
        RayDifferential ray3(r);
        ray3.time = ray2_time;
        rRec3.rayIntersect(ray3);
        ray3.mint = Epsilon;
        Float path_length3 = 0;
        path_length3 += its3.t;
        Float path_pdf3 = 1.0f;
        Spectrum path_throughput3(1.0f);
        Float G3 = 1.0f;

        Float path_pdf_1_as_1 = 1.0f;
        Float path_pdf_1_as_2 = 1.0f;
        Float path_pdf_1_as_3 = 1.0f;
        Float path_pdf_2_as_1 = 1.0f;
        Float path_pdf_2_as_2 = 1.0f;
        Float path_pdf_2_as_3 = 1.0f;
        Float path_pdf_3_as_1 = 1.0f;
        Float path_pdf_3_as_2 = 1.0f;
        Float path_pdf_3_as_3 = 1.0f;

        // bool consistency = (dot(its2.shFrame.n, its.shFrame.n) > 0.9) && (std::abs(its2.t - its.t) < 0.1);
        path2_blocked = (!its2.isValid());// || (!consistency);
        path3_blocked = (!its3.isValid());// || (!consistency);
        
        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                break;
            }

            if (path2_blocked) {
                path_throughput2 *= 0.0f;
                its2 = its; // dummy
            }

            if (path3_blocked) {
                path_throughput3 *= 0.0f;
                its3 = its; // dummy
            }

            const BSDF *bsdf = its.getBSDF(ray);
            const BSDF *bsdf2 = its2.getBSDF(ray2);
            const BSDF *bsdf3 = its3.getBSDF(ray3);

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
                    * Frame::cosTheta(its.wi) >= 0)) {
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */
            DirectSamplingRecord dRec(its);
            DirectSamplingRecord dRec2(its2);
            DirectSamplingRecord dRec3(its3);

            Spectrum neePathThroughput1 = Spectrum(0.0f);
            Spectrum neePathThroughput2 = Spectrum(0.0f);
            Spectrum neePathThroughput3 = Spectrum(0.0f);

            // path 1
            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                if (!value.isZero()) {
                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                            || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        Float em_path_length = path_length + dRec.dist;

                        Float em_modulation_weight = evalModulationWeight(ray.time, em_path_length);

                        neePathThroughput1 = path_throughput * bsdfVal * value * em_modulation_weight;
                    }
                }
            }

            // path 2
            if (!path2_blocked && rRec2.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf2->getType() & BSDF::ESmooth)) {
                Spectrum value2 = scene->sampleEmitterDirect(dRec2, rRec2.nextSample2D());
                const Emitter *emitter2 = static_cast<const Emitter *>(dRec2.object);
                if (!value2.isZero()) {
                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec2(its2, its2.toLocal(dRec2.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal2 = bsdf2->eval(bRec2);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal2.isZero() && (!m_strictNormals
                            || dot(its2.geoFrame.n, dRec2.d) * Frame::cosTheta(bRec2.wo) > 0)) {

                        Float em_path_length2 = path_length2 + dRec2.dist;

                        Float em_modulation_weight2 = evalModulationWeight(ray2.time, em_path_length2);

                        neePathThroughput2 = path_throughput2 * bsdfVal2 * value2 * em_modulation_weight2;
                    }
                }
            }

            // path 3
            if (!path3_blocked && rRec3.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf3->getType() & BSDF::ESmooth)) {
                Spectrum value3 = scene->sampleEmitterDirect(dRec3, rRec3.nextSample2D());
                const Emitter *emitter3 = static_cast<const Emitter *>(dRec3.object);
                if (!value3.isZero()) {
                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec3(its3, its3.toLocal(dRec3.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal3 = bsdf3->eval(bRec3);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal3.isZero() && (!m_strictNormals
                            || dot(its3.geoFrame.n, dRec3.d) * Frame::cosTheta(bRec3.wo) > 0)) {

                        Float em_path_length3 = path_length3 + dRec3.dist;

                        Float em_modulation_weight3 = evalModulationWeight(ray3.time, em_path_length3);

                        neePathThroughput3 = path_throughput3 * bsdfVal3 * value3 * em_modulation_weight3;
                    }
                }
            }

            // Float mis_position = miWeight(path_pdf_2_as_2, path_pdf_2_as_3, 1);
            // Float mis_direction = miWeight(path_pdf_3_as_3, path_pdf_3_as_2, 1);

            // //printf("PDF path_pdf_2_as_2 %f, path_pdf_2_as_3 %f \n", path_pdf_2_as_2, path_pdf_2_as_3);
            // //printf("PDF path_pdf_3_as_3 %f, path_pdf_3_as_2 %f \n", path_pdf_3_as_3, path_pdf_3_as_2);
            
            // //printf("MIS WEIGHT w_p : %f, w_d : %f, sum : %f \n", mis_position, mis_direction, mis_position + mis_direction);

            // if(m_preserve_primal_mis_weight_to_one){
            //     // Li += 0.5f * neePathThroughput1 / path_pdf_1_as_1;
            //     Li += 0.5f * neePathThroughput1 / path_pdf_1_as_1 * miWeight(path_pdf_1_as_1, path_pdf_1_as_2, 1);
            //     Li += 0.5f * neePathThroughput1 / path_pdf_1_as_1 * miWeight(path_pdf_1_as_1, path_pdf_1_as_3, 1);
            // } else {
            //     Li += 0.5f * mis_position * neePathThroughput1 / path_pdf_1_as_1;// * miWeight3(path_pdf_1_as_1, path_pdf_1_as_2, 0);
            //     Li += 0.5f * mis_direction * neePathThroughput1 / path_pdf_1_as_1;// * miWeight3(path_pdf_1_as_1, path_pdf_1_as_2, 0);
            // }
            
            // //Li += 0.5f * mis_position * neePathThroughput2 / path_pdf_2_as_2;// * miWeight3(path_pdf_2_as_2, path_pdf_2_as_1, 0);
            // //Li += 0.5f * mis_direction * neePathThroughput3 / path_pdf_3_as_3;// * miWeight3(path_pdf_3_as_3, path_pdf_3_as_1, 0);
            
            // Li += mis_position * neePathThroughput2 / path_pdf_2_as_2 * miWeight(path_pdf_2_as_2, path_pdf_2_as_1, 1);
            // Li += mis_direction * neePathThroughput3 / path_pdf_3_as_3 * miWeight(path_pdf_3_as_3, path_pdf_3_as_1, 1);

            if(strcmp(m_mis_method.c_str(), "mis_all") == 0){
                Li += miWeight4(path_pdf_1_as_1, path_pdf_1_as_1, path_pdf_1_as_2, path_pdf_1_as_3) * neePathThroughput1 / path_pdf_1_as_1;
                Li += miWeight4(path_pdf_1_as_1, path_pdf_1_as_1, path_pdf_1_as_2, path_pdf_1_as_3) * neePathThroughput1 / path_pdf_1_as_1;
                Li += miWeight4(path_pdf_2_as_2, path_pdf_2_as_1, path_pdf_2_as_1, path_pdf_2_as_3) * neePathThroughput2 / path_pdf_2_as_2;
                Li += miWeight4(path_pdf_3_as_3, path_pdf_3_as_1, path_pdf_3_as_1, path_pdf_3_as_2) * neePathThroughput3 / path_pdf_3_as_3;
            } else if(strcmp(m_mis_method.c_str(), "mis_separate") == 0){
                Li += 0.5 * miWeight(path_pdf_1_as_1, path_pdf_1_as_2) * neePathThroughput1 / path_pdf_1_as_1;
                Li += 0.5 * miWeight(path_pdf_1_as_1, path_pdf_1_as_3) * neePathThroughput1 / path_pdf_1_as_1;
                Li += miWeight(path_pdf_2_as_2, path_pdf_2_as_3) * miWeight(path_pdf_2_as_2, path_pdf_2_as_1) * neePathThroughput2 / path_pdf_2_as_2;
                Li += miWeight(path_pdf_3_as_3, path_pdf_3_as_2) * miWeight(path_pdf_3_as_3, path_pdf_3_as_1) * neePathThroughput3 / path_pdf_3_as_3;
            } else if(strcmp(m_mis_method.c_str(), "mis_separate2") == 0){
                Li += miWeight(path_pdf_2_as_2, path_pdf_2_as_3) * miWeight(path_pdf_1_as_1, path_pdf_1_as_2) * neePathThroughput1 / path_pdf_1_as_1;
                Li += miWeight(path_pdf_3_as_3, path_pdf_3_as_2) * miWeight(path_pdf_1_as_1, path_pdf_1_as_3) * neePathThroughput1 / path_pdf_1_as_1;
                Li += miWeight(path_pdf_2_as_2, path_pdf_2_as_3) * miWeight(path_pdf_2_as_2, path_pdf_2_as_1) * neePathThroughput2 / path_pdf_2_as_2;
                Li += miWeight(path_pdf_3_as_3, path_pdf_3_as_2) * miWeight(path_pdf_3_as_3, path_pdf_3_as_1) * neePathThroughput3 / path_pdf_3_as_3;
            }

            
            if(rRec.depth + 1>=m_maxDepth){
                break;
            }
            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            // path 1
            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            // rRec.sampler->saveState();
            Point2 usedSample = rRec.nextSample2D();
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, usedSample);
            Spectrum bsdfVal = bsdfWeight * bsdfPdf;

            if (bsdfWeight.isZero())
                break;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            bool hitEmitter = false;
            Spectrum value;
            
            Intersection its_prev = its;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);
            if (scene->rayIntersect(ray, its)) {
            } else {
                break;
            }
            G = std::abs(dot(its.shFrame.n, ray.d)) / (its.t * its.t);
            path_length += its.t;

            // path 2
            Spectrum bsdfVal2 = Spectrum(0.0f);
            Float bsdfPdf2 = 0.0f;
            Vector wo2;
            if(!path2_blocked){
                Intersection next_its = its;
                next_its.adjustTime(ray2.time);
                next_its.t = (next_its.p - its2.p).length();
                wo2 = normalize(next_its.p - its2.p);
                ray2 = Ray(its2.p, wo2, ray2.time);
                ray2.maxt = next_its.t - 1e-2;
                ray2.mint = Epsilon;
                Intersection its_temp;
                path2_blocked |= scene->rayIntersect(ray2, its_temp);

                if(!path2_blocked){
                    BSDFSamplingRecord bRec2(its2, its2.toLocal(wo2), ERadiance);
                    bsdfVal2 = bsdf2->eval(bRec2);
                    bsdfPdf2 = bsdf2->pdf(bRec2);
                    its2 = next_its;
                    G2 = std::abs(dot(its2.shFrame.n, ray2.d)) / (its2.t * its2.t);
                    path_length2 += its2.t;
                }
            }

            // path 3
            Spectrum bsdfVal3 = Spectrum(0.0f);
            Float bsdfPdf3 = 0.0f;
            if(!path3_blocked){
                //rRec.sampler->loadSavedState();
                BSDFSamplingRecord bRec3(its3, rRec3.sampler, ERadiance);
                // Point2 usedSample = rRec3.nextSample2D();
                Spectrum bsdfWeight3 = bsdf3->sample(bRec3, bsdfPdf3, usedSample);
                bsdfVal3 = bsdfWeight3 * bsdfPdf3;
                const Vector wo3 = its3.toWorld(bRec3.wo);
                ray3 = Ray(its3.p, wo3, ray3.time);
                scene->rayIntersect(ray3, its3);
                // printf("%f, %f\n", bsdfPdf, bsdfPdf3);

                // const Vector wo3 = its3.toWorld(bRec.wo);
                // ray3 = Ray(its3.p, wo3, ray3.time);
                // BSDFSamplingRecord bRec3(its3, its3.toLocal(wo3), ERadiance);
                // bsdfVal3 = bsdf3->eval(bRec3);
                // bsdfPdf3 = bsdf3->pdf(bRec3);
                // scene->rayIntersect(ray3, its3);
                path3_blocked |= (!its3.isValid());
                
                if(!path3_blocked){
                    //BSDFSamplingRecord bRec3(its3, its3.toLocal(wo3), ERadiance);
                    //bsdfVal3 = bsdf3->eval(bRec3);
                    //bsdfPdf3 = bsdf3->pdf(bRec3);
                    path_length3 += its3.t;

                    if(!(bRec3.sampledType & BSDF::EDelta)){
                        G3 = std::abs(dot(its3.shFrame.n, ray3.d)) / (its3.t * its3.t);
                    } else {
                        G3 = 1.0f;
                    }
                    
                    //printf("%f, %f\n", bsdfPdf, bsdfPdf3);
                } else {
                    bsdfVal3 *= 0.0f;
                    bsdfPdf3 *= 0.0f;
                }
            }
            
            // path 3 as 2
            Float bsdfPdf3_as_2 = 0.0f;
            Float G3_as_2 = 0.0f;
            if(!path3_blocked){
                Intersection its_3_as_2 = its3;
                its_3_as_2.adjustTime(ray.time);
                const Vector wo_3_as_2 = normalize(its_3_as_2.p - its_prev.p);
                BSDFSamplingRecord bRec3_as_2(its_prev, its_prev.toLocal(wo_3_as_2), ERadiance);
                // Spectrum bsdfVal3_as_2 = bsdf->eval(bRec3_as_2);
                bsdfPdf3_as_2 = bsdf->pdf(bRec3_as_2);
                G3_as_2 = std::abs(dot(its_3_as_2.shFrame.n, wo_3_as_2)) / (its_3_as_2.p - its_prev.p).lengthSquared();
            }

            // path 2 as 3
            Float bsdfPdf2_as_3 = 0.0f;
            if(!path2_blocked){
                BSDFSamplingRecord bRec2_as_3(its_prev, its_prev.toLocal(wo2), ERadiance);
                // Spectrum bsdfVal2_as_3 = bsdf->eval(bRec2_as_3);
                bsdfPdf2_as_3 = bsdf->pdf(bRec2_as_3);
                //printf("bsdfPdf2_as_3 %f\n", bsdfPdf2_as_3);
            }
            
            path_pdf_1_as_1 *= bsdfPdf * G; 
            path_pdf_1_as_2 *= bsdfPdf2 * G2;
            path_pdf_1_as_3 *= bsdfPdf3 * G;
            
            path_pdf_2_as_1 *= bsdfPdf2 * G2;
            path_pdf_2_as_2 *= bsdfPdf * G;
            path_pdf_2_as_3 *= bsdfPdf2_as_3 * G2;
            
            path_pdf_3_as_1 *= bsdfPdf3 * G3;
            path_pdf_3_as_2 *= bsdfPdf3_as_2 * G3_as_2;
            path_pdf_3_as_3 *= bsdfPdf * G3;

            path_throughput *= bsdfVal * G;
            path_throughput2 *= bsdfVal2 * G2;
            path_throughput3 *= bsdfVal3 * G3;

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec2.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec3.type = RadianceQueryRecord::ERadianceNoEmission;
            
            rRec.depth++;
        }

        return Li;
    }

    Spectrum Li_path_sampler_correlation(const RayDifferential &r, RadianceQueryRecord &_rRec) const
    {
        RadianceQueryRecord rRec = _rRec;
        int n_antithetic = this->m_antithetic_shifts.size();
        rRec.sampler->saveState();

        // primal path
        RayDifferential ray(r);
        Spectrum Li = Li_single(ray, rRec);

        // antithetic paths
        for(float antithetic_shift : this->m_antithetic_shifts){
            
            // RayDifferential ray(r);
            // ray.time = r.time + antithetic_shift * m_time;
            // ray.time = std::fmod(ray.time, m_time);
            RadianceQueryRecord rRec = _rRec;
            rRec.sampler->loadSavedState();
            RayDifferential sensorRay;
            rRec.scene->getSensor()->sampleRayDifferential(
                sensorRay, _rRec.samplePos, _rRec.apertureSample, std::fmod(_rRec.timeSample + antithetic_shift, 1.0));
            sensorRay.scaleDifferential(_rRec.diffScaleFactor);

            Li += Li_single(sensorRay, rRec);
        }
        for(float antithetic_shift : this->m_antithetic_shifts){
            rRec.sampler->advance();
        }
        return Li * (1.0) / (n_antithetic + 1);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        // path sampler correlation
        if(strcmp(m_antithetic_sampling_mode.c_str(), "sampler_correlation")==0){
            return Li_path_sampler_correlation(r, rRec);
        }

        // antithetic time setting
        Float ray2_time = r.time + 0.5 * m_time;

        if(ray2_time >= m_time){
            ray2_time -= m_time;
        }
        if(!m_antithetic_sampling_by_shift){
            ray2_time = m_time - r.time;
        }

        // trace two antithetic path
        if(strcmp(m_antithetic_sampling_mode.c_str(), "mis")==0){
            return Li_helper_mis_new(r, rRec, 0.5);
        } 
        // trace only one antithetic path
        else {
            return Li_helper(r, rRec, 0.5);
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
    bool m_is_objects_transformed_for_time;
    bool m_force_constant_attenuation;
    bool m_correlate_time_samples;
    bool m_antithetic_sampling_by_shift;
    bool m_preserve_primal_mis_weight_to_one;
    float m_primal_antithetic_mis_power;

    std::string m_mis_method;
    std::string m_antithetic_sampling_mode;
};

MTS_IMPLEMENT_CLASS_S(ToFAntitheticPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ToFAntitheticPathTracer, "ToF antithetic path tracer");
MTS_NAMESPACE_END


