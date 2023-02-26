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

        m_low_frequency_component_only = props.getBoolean("low_frequency_component_only", false);
        m_is_objects_transformed_for_time = props.getBoolean("is_object_transformed_for_time", false);
        m_force_constant_attenuation = props.getBoolean("force_constant_attenuation", false);
        m_antithetic_sampling_by_shift = props.getBoolean("antitheticSamplingByShift", true);
        m_preserve_primal_mis_weight_to_one = props.getBoolean("preservePrimalMISWeightToOne", true);
        m_mis_method = props.getString("misMethod", "mix_all");

        m_antithetic_sampling_mode = props.getString("antitheticSamplingMode", "position");
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

    void accumulate_nee_Li(Spectrum &Li, PathTracePart &path1, PathTracePart &path2) const{
        if(path1.path_pdf_nee_as_nee > 0){
            Float p1 = path1.path_pdf_as(path1)* path1.path_pdf_nee_as_nee;
            Float p2 = path1.path_pdf_as(path1) * path1.path_pdf_nee_as_bsdf;
            Float p3 = path1.path_pdf_as(path2) * path2.path_pdf_nee_as_nee;
            Float p4 = path1.path_pdf_as(path2) * path2.path_pdf_nee_as_bsdf;
            Float modulation = evalModulationWeight(path1.ray.time, path1.em_path_length);
            Li += modulation * path1.path_throughput_nee / p1 * miWeight4(p1, p2, p3, p4);
        }
    }

    void accumulate_bsdf_Li(Spectrum &Li, PathTracePart &path1, PathTracePart &path2) const{
        if(path1.hitEmitter){
            Float p1 = path1.path_pdf_as(path1);
            Float p2 = path1.path_pdf_as(path2);
            Float p3 = path1.path_pdf_as_nee(path1);
            Float p4 = path1.path_pdf_as_nee(path2);
            Float modulation = evalModulationWeight(path1.ray.time, path1.path_length);
            Li += modulation * path1.path_throughput * path1.hitEmitterValue / p1 * miWeight4(p1, p2, p3, p4);
        }
    }


    Spectrum Li_helper(const RayDifferential &r, RadianceQueryRecord &rRec, Float ray2_time) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Spectrum Li(0.0f);
        RadianceQueryRecord rRec2 = rRec;

        // path 1
        //Intersection &its1 = rRec.its;
        RayDifferential ray1(r);
        rRec.rayIntersect(ray1);
        PathTracePart path1(ray1, rRec, 0);

        // path 2
        //Intersection &its2 = rRec2.its;
        RayDifferential ray2(r);
        ray2.time = ray2_time;
        rRec2.rayIntersect(ray2);
        PathTracePart path2(ray2, rRec2, 1);

        // bool path2_blocked = (!path2.its.isValid());
        while(rRec.depth <= m_maxDepth || m_maxDepth < 0){
            if(!path1.its.isValid() || path1.path_terminated){
                break;
            }

            if(!path2.its.isValid()){
                path2.set_path_pdf_as(path2, 0.0);
                path2.path_terminated = true;
            }

            if(rRec.depth == 0 && !m_hideEmitters){
                accumulate_bsdf_Li(Li, path1, path2);
                accumulate_bsdf_Li(Li, path2, path1);
            }
            
            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(path1.ray.d, path1.its.geoFrame.n)
                    * Frame::cosTheta(path1.its.wi) >= 0)) {
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */
            Point2 nee_sample = rRec.nextSample2D();
            

            path1.nee_trace(nee_sample);
            path2.nee_trace(nee_sample);
            
            // Accumulate nee Li
            accumulate_nee_Li(Li, path1, path2);
            accumulate_nee_Li(Li, path2, path1);
            

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
            
            bool use_positional_correlation = (strcmp(m_antithetic_sampling_mode.c_str(), "position") == 0);

            if(use_positional_correlation){
                path2.get_next_ray_from_its(path1.its);
                PathTracePart::calculate_bsdf_pdf(path1, path2, 1);
            } else {
                path2.get_next_ray_from_sample(bsdf_sample);
                PathTracePart::calculate_bsdf_pdf(path1, path2, path1.G / path2.G);
            }
            
            accumulate_bsdf_Li(Li, path1, path2);
            accumulate_bsdf_Li(Li, path2, path1);
            // break;

            //rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            //rRec2.type = RadianceQueryRecord::ERadianceNoEmission;
            
            rRec.depth++;
            rRec2.depth++;
        }
        return Li;
    }

    Spectrum Li_helper_mis_new(const RayDifferential &r, RadianceQueryRecord &rRec, Float ray2_time) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Spectrum Li(0.0f);
        RadianceQueryRecord rRec2 = rRec;
        RadianceQueryRecord rRec3 = rRec;
        RadianceQueryRecord rRec4 = rRec;
        RadianceQueryRecord rRec5 = rRec;

        // path 1
        RayDifferential ray1(r);
        rRec.rayIntersect(ray1);
        PathTracePart path1(ray1, rRec, 0);

        // path 2 (positional correlation)
        RayDifferential ray2(r);
        ray2.time = ray2_time;
        rRec2.rayIntersect(ray2);
        PathTracePart path2(ray2, rRec2, 1);

        // path 3 (sampler correlation)
        RayDifferential ray3(r);
        ray3.time = ray2_time;
        rRec3.rayIntersect(ray3);
        PathTracePart path3(ray3, rRec3, 2);
        
        // path 4 (positional correlation as directional correlation)
        RayDifferential ray4(r);
        rRec4.rayIntersect(ray4);
        PathTracePart path4(ray4, rRec4, 3);

        // path 5 (directional correlation as positional correlation)
        RayDifferential ray5(r);
        rRec5.rayIntersect(ray5);
        PathTracePart path5(ray5, rRec5, 4);

        while(rRec.depth <= m_maxDepth || m_maxDepth < 0){
            if(!path1.its.isValid() || path1.path_terminated){
                break;
            }

            if(!path2.its.isValid()){
                path2.set_path_pdf_as(path2, 0.0);
                path2.path_terminated = true;
            }

            if(!path3.its.isValid()){
                path3.set_path_pdf_as(path3, 0.0);
                path3.path_terminated = true;
            }

            if(!path4.its.isValid()){
                path4.set_path_pdf_as(path4, 0.0);
                path4.path_terminated = true;
            }

            if(!path5.its.isValid()){
                path5.set_path_pdf_as(path5, 0.0);
                path5.path_terminated = true;
            }

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(path1.ray.d, path1.its.geoFrame.n)
                    * Frame::cosTheta(path1.its.wi) >= 0)) {
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */
            Point2 nee_sample = rRec.nextSample2D();
            
            path1.nee_trace(nee_sample);
            path2.nee_trace(nee_sample);
            path3.nee_trace(nee_sample);

            // Accumulate nee Li
            accumulate_nee_Li(Li, path1, path2);
            accumulate_nee_Li(Li, path2, path1);

            if(rRec.depth + 1>=m_maxDepth){
                break;
            }

             /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            Point2 bsdf_sample = rRec.nextSample2D();
            path1.get_next_ray_from_sample(bsdf_sample);
            
            path2.get_next_ray_from_its(path1.its);
            PathTracePart::calculate_bsdf_pdf(path1, path2, 1);
            
            path3.get_next_ray_from_sample(bsdf_sample);
            PathTracePart::calculate_bsdf_pdf(path1, path3, path1.G / path3.G);

            path4.get_next_ray_from_sample(path2.sample);
            PathTracePart::calculate_bsdf_pdf_as_other(path4, path2, path3, path4.G / path2.G);

            path5.get_next_ray_from_its(path3.its);
            PathTracePart::calculate_bsdf_pdf_as_other(path5, path3, path2, 1);

            // PathTracePart::calculate_bsdf_pdf_as_other(path5, path3, 1);

            //path2.set_path_pdf_as(path3, );

        }
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

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        // antithetic time setting
        Float ray2_time = r.time + m_antithetic_shift * m_time;

        if(ray2_time >= m_time){
            ray2_time -= m_time;
        }
        if(!m_antithetic_sampling_by_shift){
            ray2_time = m_time - r.time;
        }

        // trace two antithetic path
        if(strcmp(m_antithetic_sampling_mode.c_str(), "mis")==0){
            return Li_helper_mis(r, rRec, ray2_time);
        } 
        // trace only one antithetic path
        else {
            return Li_helper(r, rRec, ray2_time);
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
    
    bool m_low_frequency_component_only;
    bool m_is_objects_transformed_for_time;
    bool m_force_constant_attenuation;
    bool m_correlate_time_samples;
    bool m_antithetic_sampling_by_shift;
    bool m_preserve_primal_mis_weight_to_one;
    std::string m_mis_method;
    std::string m_antithetic_sampling_mode;
};

MTS_IMPLEMENT_CLASS_S(ToFAntitheticPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ToFAntitheticPathTracer, "ToF antithetic path tracer");
MTS_NAMESPACE_END


