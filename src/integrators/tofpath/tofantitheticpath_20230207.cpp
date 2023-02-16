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

        m_low_frequency_component_only = props.getBoolean("low_frequency_component_only", false);
        m_is_objects_transformed_for_time = props.getBoolean("is_object_transformed_for_time", false);
        m_force_constant_attenuation = props.getBoolean("force_constant_attenuation", false);
        m_antithetic_sampling_by_shift = props.getBoolean("antitheticSamplingByShift", true);
        m_preserve_primal_mis_weight_to_one = props.getBoolean("preservePrimalMISWeightToOne", true);

        m_antithetic_sampling_mode = props.getString("antitheticSamplingMode", "position");
    }

    /// Unserialize from a binary data stream
    ToFAntitheticPathTracer(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    Float evalModulationWeight(Float &ray_time, Float &path_length) const{
        Float w_g = 2 * M_PI * m_illumination_modulation_frequency_mhz * 1e6;
        Float w_f = 2 * M_PI * m_sensor_modulation_frequency_mhz * 1e6;
        Float delta_t = path_length / (3e8);
        Float phi = (2 * M_PI * m_illumination_modulation_frequency_mhz) / 300 * path_length;

        if(m_low_frequency_component_only){
            Float fg_t = 0.25 * std::cos((w_f - w_g) * ray_time + phi);
            return fg_t;
        } 
        
        Float g_t = 0.5 * std::cos(w_g * ray_time - phi) + 0.5;
        Float f_t = std::cos(w_f * ray_time);
        return f_t * g_t;
    }

    bool check_consistency(Intersection& its1, Intersection& its2) const{
        bool consistent = its1.isValid() && its2.isValid() && (its1.p - its2.p).length() < 0.1 && dot(its1.shFrame.n, its2.shFrame.n) > 0.9;
        return consistent;
    }

    Spectrum Li_helper(const RayDifferential &r, RadianceQueryRecord &rRec, Float ray2_time) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Spectrum Li(0.0f);
        RadianceQueryRecord rRec2 = rRec;

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

        Float path_pdf_1_as_1 = 1.0f;
        Float path_pdf_1_as_2 = 1.0f;
        Float path_pdf_2_as_1 = 1.0f;
        Float path_pdf_2_as_2 = 1.0f;

        // bool consistency = (dot(its2.shFrame.n, its.shFrame.n) > 0.9) && (std::abs(its2.t - its.t) < 0.1);
        path2_blocked = (!its2.isValid());

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                break;
            }

            if (path2_blocked) {
                path_throughput2 *= 0.0f;
                its2 = its; // dummy
            }

            const BSDF *bsdf = its.getBSDF(ray);
            const BSDF *bsdf2 = its2.getBSDF(ray2);

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

            Float neePathBSDFPdf = 0.0f;
            Float neePathNEEPdf = 0.0f;
            Spectrum neePathThroughput = Spectrum(0.0f);
            Float neePathBSDFPdf2 = 0.0f;
            Float neePathNEEPdf2 = 0.0f;
            Spectrum neePathThroughput2 = Spectrum(0.0f);

            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                value *= dRec.pdf;

                ///////////////////////
                // path 1
                ///////////////////////
                const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                if (!value.isZero()) {
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

                        if(emitter->isOnSurface()){
                            Float G = std::abs(dot(dRec.d, dRec.n)) / (dRec.dist * dRec.dist);
                            neePathNEEPdf = dRec.pdf * G;
                            neePathBSDFPdf = bsdfPdf * G;
                            neePathThroughput = path_throughput * bsdfVal * G *  value * em_modulation_weight;
                        } else {
                            neePathNEEPdf = dRec.pdf;
                            neePathBSDFPdf = 0;
                            neePathThroughput = path_throughput * bsdfVal * value * em_modulation_weight;
                        }
                    }
                }

                ///////////////////////
                // path 2
                // (Sampler Correlation = Position Correlation)
                ///////////////////////

                if(!path2_blocked && ! value.isZero()){
                    Spectrum value2 = value;
                    dRec2.p = dRec.p;
                    dRec2.n = dRec.n;
                    dRec2.d = normalize(dRec2.p - its2.p);
                    dRec2.dist = (dRec2.p - its2.p).length();
                    dRec2.measure = dRec.measure;
                    dRec2.object = dRec.object;
                    dRec2.uv = dRec.uv;

                    if(emitter->isOnSurface()){
                        dRec2.pdf = dRec.pdf;
                        if(dRec.pdf != 0){
                            Float G = std::abs(dot(dRec.d, dRec.n)) / (dRec.dist * dRec.dist);
                            Float G2 = std::abs(dot(dRec2.d, dRec2.n)) / (dRec2.dist * dRec2.dist);
                            if(G2 > 0){
                                dRec2.pdf = dRec.pdf * G / G2;
                            }    
                        }
                    } else {
                        dRec2.pdf = dRec.pdf;
                        Float G = 1 / (dRec.dist * dRec.dist);
                        Float G2 = 1 / (dRec2.dist * dRec2.dist);
                        value2 = value / G * G2;
                    }

                    Ray shadowray = Ray(its2.p, dRec2.d, ray2.time);
                    shadowray.maxt = dRec2.dist - 1e-2;
                    shadowray.mint = Epsilon;
                    Intersection its_temp;
                    bool em_path2_blocked = scene->rayIntersect(shadowray, its_temp);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec2(its2, its2.toLocal(dRec2.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal2 = bsdf2->eval(bRec2);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!em_path2_blocked && !bsdfVal2.isZero() && (!m_strictNormals
                            || dot(its2.geoFrame.n, dRec2.d) * Frame::cosTheta(bRec2.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                        using BSDF sampling */
                        Float bsdfPdf2 = (emitter->isOnSurface() && dRec2.measure == ESolidAngle)
                            ? bsdf2->pdf(bRec2) : 0;

                        Float em_path_length2 = path_length2 + dRec2.dist;

                        Float em_modulation_weight2 = evalModulationWeight(ray2.time, em_path_length2);

                        if(emitter->isOnSurface()){
                            Float G2 = std::abs(dot(dRec2.d, dRec2.n)) / (dRec2.dist * dRec2.dist);
                            neePathNEEPdf2 = dRec2.pdf * G2;
                            neePathBSDFPdf2 = bsdfPdf2 * G2;
                            neePathThroughput2 = path_throughput2 * bsdfVal2 * G2 *  value2 * em_modulation_weight2;
                        } else {
                            neePathNEEPdf2 = dRec2.pdf;
                            neePathBSDFPdf2 = 0;
                            neePathThroughput2 = path_throughput2 * bsdfVal2 * value2 * em_modulation_weight2;
                        }
                    }
                }
            }

            // MIS for NEE 
            if(neePathNEEPdf > 0.0){
                Float p1 = path_pdf_1_as_1 * neePathNEEPdf;
                Float p2 = path_pdf_1_as_1 * neePathBSDFPdf;
                Float p3 = path_pdf_1_as_2 * neePathNEEPdf2;
                Float p4 = path_pdf_1_as_2 * neePathBSDFPdf2;
                Li += neePathThroughput / p1 * miWeight4(p1, p2, p3, p4);
            }
            if(neePathNEEPdf > 0.0){
                Float p1 = path_pdf_2_as_2 * neePathNEEPdf;
                Float p2 = path_pdf_2_as_2 * neePathBSDFPdf;
                Float p3 = path_pdf_2_as_1 * neePathNEEPdf2;
                Float p4 = path_pdf_2_as_1 * neePathBSDFPdf2;
                Li += neePathThroughput2 / p1 * miWeight4(p1, p2, p3, p4);
            }

            if(rRec.depth + 1>=m_maxDepth){
                break;
            }
            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            // Strategy choosing heuristics
            Float roughness = bsdf->getRoughness(its, 0);
            roughness = std::min(roughness, 20.0);
            Float reflected_dist = -1.0f;
            Vector3 reflected_dir = reflect(-ray.d, its.shFrame.n);
            Ray ray_mirror = Ray(its.p, reflected_dir, ray.time);
            Intersection its_mirror;
            Float reflected_cos_o = 1.0f;
            if(scene->rayIntersect(ray_mirror, its_mirror)){
                reflected_dist = its_mirror.t;
                reflected_cos_o = std::abs(dot(its_mirror.shFrame.n, ray_mirror.d));
            } else {
                reflected_dist = 10.0f;
                reflected_cos_o = 1.0f;
            }

            Float bsdfPdf;
            Spectrum bsdfVal = Spectrum(0.0f);
            Float bsdfPdf2 = 0.0f;
            Spectrum bsdfVal2 = Spectrum(0.0f);
            Float lumPdf = 0.0;
            Float lumPdf2 = 0.0;
            Float modulation_weight = 0.0;
            Float modulation_weight2 = 0.0;
            Float path_pdf_1_as_1_nee = 0.0;
            Float path_pdf_1_as_2_nee = 0.0;
            Float path_pdf_2_as_1_nee = 0.0;
            Float path_pdf_2_as_2_nee = 0.0;
            Float path_pdf_1_as_1_bsdf = 0.0;
            Float path_pdf_1_as_2_bsdf = 0.0;
            Float path_pdf_2_as_1_bsdf = 0.0;
            Float path_pdf_2_as_2_bsdf = 0.0;
            bool hitEmitter = false;
            Spectrum value = Spectrum(0.0f);
            bool hitEmitter2 = false;
            Spectrum value2 = Spectrum(0.0f);
            
            ////////////////////////////////////
            // path 1
            ////////////////////////////////////

            /* Sample BSDF * cos(theta) */
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Point2 usedSample = rRec.nextSample2D();
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, usedSample);
            bsdfVal = bsdfWeight * bsdfPdf;

            if (bsdfWeight.isZero())
                break;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

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
                break;
            }

            //if(!(bRec.sampledType & BSDF::EDelta)){
                G = std::abs(dot(its.shFrame.n, ray.d)) / (its.t * its.t);
            //} else {
            //    G = 1.0;
            //}
            path_length += its.t;

            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                modulation_weight = evalModulationWeight(ray.time, path_length);
            }

            path_pdf_1_as_1_nee = path_pdf_1_as_1 * lumPdf * G;
            path_pdf_1_as_1_bsdf = path_pdf_1_as_1 * bsdfPdf * G; 

            ////////////////////////////////////
            // path 2
            ////////////////////////////////////

            //bool force_position = (strcmp(m_antithetic_sampling_mode.c_str(), "position") == 0);
            //bool force_direction = (strcmp(m_antithetic_sampling_mode.c_str(), "direction") == 0);
            bool useMixing = (strcmp(m_antithetic_sampling_mode.c_str(), "mixed2") == 0);
            
            if(!path2_blocked && useMixing){
                // Mixed position
                
                // (1) Positional Correlation
                Intersection next_its_p = its;
                next_its_p.adjustTime(ray2.time);
                next_its_p.t = (next_its_p.p - its2.p).length();

                // (2) Directional Correlation
                Intersection next_its_d;
                BSDFSamplingRecord bRec2_d(its2, rRec2.sampler, ERadiance);
                Spectrum bsdfWeight_d = bsdf2->sample(bRec2_d, bsdfPdf2, usedSample);
                const Vector wo2_d = its2.toWorld(bRec2_d.wo);
                Ray ray2_d = Ray(its2.p, wo2_d, ray2.time);

                std::string shape_name = typeid(*its.shape).name();
                // std::cout <<  "NAME!!!"<< typeid(*its.shape).name() << std::endl;
                
                Float temp[2];
                Float t_temp;
                Float t_area;
                Float t_dist;
                Float t_cos;
                bool edge_continuous[3];

                // next_its_p.shape->adjustTime(ray2.time);
                if(next_its_p.instance){
                    const Instance *instance = reinterpret_cast<const Instance*>(next_its_p.instance);
                    Ray ray_temp;
                    const Transform &trafo = instance->getAnimatedTransform()->eval(ray2.time);
                    trafo.inverse()(ray2_d, ray_temp);

                    //std::cout << "TRAFO" << trafo.toString() << std::endl;
                    //Point p0 = (trafo1.inverse()(its.p0));
                    //Point p1 = (trafo1.inverse()(its.p1));
                    //Point p2 = (trafo1.inverse()(its.p2));
                    
                    if (shape_name.find("Rectangle") != std::string::npos){
                        next_its_p.shape->rayIntersectForced(ray_temp, ray_temp.mint, ray_temp.maxt, t_temp, temp);
                        next_its_d.p = trafo(ray_temp(t_temp));
                        t_area = 1.0;
                        t_dist = 1.0;
                        t_cos = 1.0;
                    } else {
                        Triangle::rayIntersectForced(its.p0, its.p1, its.p2, ray_temp, temp[0], temp[1], t_temp);
                        next_its_d.p = trafo(ray_temp(t_temp));
                        if(t_temp < 0){
                            //std::cout << next_its_d.p.toString() << std::endl;
                            //std::cout << its.p1.toString() << std::endl;
                            //std::cout << its.p2.toString() << std::endl;
                        }
                        const Transform &trafo1 = instance->getAnimatedTransform()->eval(ray.time);
                        Point p0 = trafo1(its.p0);
                        Point p1 = trafo1(its.p1);
                        Point p2 = trafo1(its.p2);
                        Point pa0 = trafo(its.p0);
                        Point pa1 = trafo(its.p1);
                        Point pa2 = trafo(its.p2);
                        edge_continuous[0] = its.e0;
                        edge_continuous[1] = its.e1;
                        edge_continuous[2] = its.e2;
                        Vector e1 = p1 - p0;
                        Vector e2 = p2 - p0;
                        t_area = cross(e1, e2).length();
                        t_dist = (its2.p - (p0 + p1 + p2) / 3).length();
                        t_cos = dot(normalize(its2.p - (p0 + p1 + p2) / 3), its.shFrame.n);
                        t_cos = std::abs(t_cos);
                    }
                    //trafo.inverse()(_ray, ray);
                } else {
                    next_its_p.shape->rayIntersect(ray2_d, ray2_d.mint, ray2_d.maxt, t_temp, temp);
                    next_its_d.p = ray2_d(t_temp);
                    t_area = 1.0;
                    t_dist = 1.0;
                    t_cos = 1.0;                    
                }

                Intersection next_its_d_temp;
                ray2_d = Ray(its2.p, wo2_d, ray2.time);
                scene->rayIntersect(ray2_d, next_its_d_temp);
                if(next_its_d_temp.isValid()){
                    if((next_its_d.p - next_its_d_temp.p).length() > 0.1){
                        // printf("ours, (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f), (%.3f, %.3f)\n", 
                        // next_its_d.p.x, next_its_d.p.y, next_its_d.p.z,
                        // next_its_d_temp.p.x, next_its_d_temp.p.y, next_its_d_temp.p.z,
                        // next_its_d.uv[0], next_its_d.uv[1]
                        // );

                        // printf("Distance T: %f, %f \n", t_temp, next_its_d_temp.t);

                        // std::cout << next_its_d.p.toString() << next_its_d_temp.p.toString() << std::endl;
                    }
                }

                //Point p_temp = its2.p + next_its_d_temp.t * wo2_d;
                
                //std::cout << "ITS" << its.shFrame.toString() << std::endl;

                // if(next_its_d_temp.isValid()){
                //     printf("ours, (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f), (%.3f, %.3f)\n", 
                //     next_its_d.p.x, next_its_d.p.y, next_its_d.p.z,
                //     next_its_d_temp.p.x, next_its_d_temp.p.y, next_its_d_temp.p.z,
                //     next_its_d.uv[0], next_its_d.uv[1]
                //     );
                //     //std::cout << next_its_d_temp.p.toString() << p_temp.toString()<< std::endl;
                //     printf("Distance T: %f, %f \n", t_temp, next_its_d_temp.t);
                // }
                //->rayIntersectForced(ray2_d, ray2_d.mint, ray2_d.maxt, t_temp, temp);
                
                // next_its_d.p = ray2_d(t_temp);//its2.p + wo2_d * t_temp;
                //std::cout << "T temp: " << t_temp << std::endl;

                next_its_d.t = t_temp;
                next_its_d.uv = Point2(0.5f * (temp[0]+1), 0.5f * (temp[1]+1));
                next_its_d.barycentric = Vector(1 - temp[0] - temp[1], temp[0], temp[1]);

                //Float next_ids_d_k = dot(its2.p - next_its_p.p, next_its_p.shFrame.n) / dot(its.p - its2.p, next_its_p.shFrame.n);
                //next_its_d.p = its2.p + (its.p - its2.p);

                //printf("ours2, %f, %f, %f\n", next_its_d.p.x, next_its_d.p.y, next_its_d.p.z);
                //printf("uv, %f, %f\n", next_its_d.uv[0], next_its_d.uv[1]);
                // Check discontinuity
                // path2_blocked |= !next_its_d.isValid();//(!check_consistency(its, next_its_d)); 
                // if(!next_its_d.isValid()){
                //     printf("Coords, (%.3f, %.3f, %.3f)\n", its2.p.x, its2.p.y, its2.p.z);
                //     printf("Coords, (%.3f, %.3f, %.3f)\n", next_its_d.p.x, next_its_d.p.y, next_its_d.p.z);
                //     printf("UV Coords, (%.3f, %.3f)\n", next_its_d.uv.x, next_its_d.uv.y);
                // }    

                Float G2_d = 1.0;
                //if(!(bRec2_d.sampledType & BSDF::EDelta)){
                if(!path2_blocked){
                    Float d_temp = next_its_d.t;//std::max(next_its_d.t, 1e-5);
                    G2_d = std::abs(dot(next_its_p.shFrame.n, ray2_d.d)/(d_temp * d_temp));
                } else {
                    G2_d = 1.0;
                }

                // (3) Mixed
                Intersection next_its_m = next_its_p;

                
                Float jacobian_m = 1.0;
                Float weight_p = 0.0;
                Float weight_p_prime = 0.0;
                // Float m_k = (1 / (roughness + 1e-5) * 0.8 + 1);

                Float steradian = t_area * t_cos / (t_dist * t_dist);
                // Float area = roughness * t_dist * t_dist / (t_cos);
                // Float position_correlation_probability = 6.0f * area;
                // position_correlation_probability = std::min(position_correlation_probability, 1.0);

                // m_k large --> more directional
                // Float m_k = 1;//(t_area / (area + 1e-5) * 0.8 + 1);
                // Float m_k = (steradian * roughness * 0.2 + 1);
                Float m_k = (steradian / roughness * 0.2);
                // if(steradian < 0.01){
                //     m_k = 0.0;
                // }

                // m_k = 0.0;
                // std::cout << "Steradians" << area << std::endl;
                // std::cout << "t_dist" << t_dist << std::endl;
                // std::cout << "t_cos" << t_cos << std::endl;
                // std::cout << "t_area" << t_area << std::endl;
                
                bool mesh_hit_preserved = true;

                // Rectangle
                if (shape_name.find("Rectangle") != std::string::npos){
                    Float m_x_1 = its.uv[0];
                    Float m_x_2 = 1 - its.uv[0];
                    Float m_x_3 = its.uv[1];
                    Float m_x_4 = 1 - its.uv[1];
                    Float m_x;
                    Vector2 m_direction;
                    if((m_x_1 <= m_x_2) && (m_x_1 <= m_x_3) && (m_x_1 <= m_x_4)){
                        m_x = m_x_1;
                        m_direction = Vector2(1.0, 0.0);
                    }
                    else if((m_x_2 <= m_x_1) && (m_x_2 <= m_x_3) && (m_x_2 <= m_x_4)){
                        m_x = m_x_2;
                        m_direction = Vector2(-1.0, 0.0);
                    }
                    else if((m_x_3 <= m_x_1) && (m_x_3 <= m_x_2) && (m_x_3 <= m_x_4)){
                        m_x = m_x_3;
                        m_direction = Vector2(0.0, 1.0);
                    }
                    else if((m_x_4 <= m_x_1) && (m_x_4 <= m_x_2) && (m_x_4 <= m_x_3)){
                        m_x = m_x_4;
                        m_direction = Vector2(0.0, -1.0);
                    }
                    //m_x = m_x_1;
                    //m_direction = Vector2(1.0, 0.0);
                    
                    m_x = 1 - 2 * m_x;
                    m_x = std::abs(m_x);

                    //m_x = its.p[2];

                    weight_p = std::pow(m_x, m_k);
                    weight_p_prime = m_k * std::pow(m_x, m_k - 1);

                    Float jacobian_p_sqrt = 1;
                    Float jacobian_d_sqrt = std::sqrt(std::abs(G / G2_d));
                    
                    Float jacobian_m_1 = weight_p * jacobian_p_sqrt + (1 - weight_p) * jacobian_d_sqrt;
                    Float jacobian_m_2 = weight_p_prime * (-2) * dot(m_direction, next_its_p.uv - next_its_d.uv) + weight_p * jacobian_p_sqrt + (1 - weight_p) * jacobian_d_sqrt;
                    //Float jacobian_m = weight_p_prime * (next_its_p.p[2] - next_its_d.p[2]) + weight_p * jacobian_p_sqrt + (1 - weight_p) * jacobian_d_sqrt;
                    jacobian_m = jacobian_m_1 * jacobian_m_2;
                    // jacobian_m = std::abs(G / G2_d);//std::abs(jacobian_m);

                    Float m_u = next_its_p.uv[0] * weight_p + next_its_d.uv[0] * (1-weight_p);
                    Float m_v = next_its_p.uv[1] * weight_p + next_its_d.uv[1] * (1-weight_p);
                    // // printf("m_v, m_u, %f, %f\n", m_v, m_u);

                    if((m_u > 1) || (m_u < 0) || (m_v > 1) || (m_v < 0)){
                        path2_blocked = true;
                    }
                } 
                
                // Triangle
                else {

                    Float m_x_1 = its.barycentric[0];
                    Float m_x_2 = its.barycentric[1];
                    Float m_x_3 = its.barycentric[2];
                    // std::cout << its.p2.toString() << std::endl;

                    Float m_x;
                    Vector m_direction;
                    int N = 3;

                    if(edge_continuous[0]){
                        m_x_1 = 100;
                        N -= 1;
                    }
                    if(edge_continuous[1]){
                        m_x_2 = 100;
                        N -= 1;
                    }
                    if(edge_continuous[2]){
                        m_x_3 = 100;
                        N -= 1;
                    }

                    if((m_x_1 <= m_x_2) && (m_x_1 <= m_x_3)){
                        m_x = m_x_1;
                        m_direction = Vector(1.0, 0.0, 0.0);
                    }
                    else if((m_x_2 <= m_x_1) && (m_x_2 <= m_x_3)){
                        m_x = m_x_2;
                        m_direction = Vector(0.0, 1.0, 0.0);
                    }
                    else if((m_x_3 <= m_x_2) && (m_x_3 <= m_x_1)){
                        m_x = m_x_3;
                        m_direction = Vector(0.0, 0.0, 1.0);
                    }
                    // std::cout << "N" << N << std::endl;

                    m_x = 1 - N * m_x;
                    m_x = std::abs(m_x);
                    weight_p = 0;//std::pow(m_x, m_k);
                    weight_p_prime = 0;//m_k * std::pow(m_x, m_k - 1);
                    
                    Float jacobian_p_sqrt = 1;
                    Float jacobian_d_sqrt = std::sqrt(std::abs(G / G2_d));
                    
                    Float jacobian_m_1 = weight_p * jacobian_p_sqrt + (1 - weight_p) * jacobian_d_sqrt;
                    Float jacobian_m_2 = weight_p_prime * (-N) * dot(m_direction, next_its_p.barycentric - next_its_d.barycentric) + weight_p * jacobian_p_sqrt + (1 - weight_p) * jacobian_d_sqrt;
                    jacobian_m = jacobian_m_1 * jacobian_m_2;

                    m_x_1 = next_its_p.barycentric[0] * weight_p + next_its_d.barycentric[0] * (1-weight_p);
                    m_x_2 = next_its_p.barycentric[1] * weight_p + next_its_d.barycentric[1] * (1-weight_p);
                    m_x_3 = next_its_p.barycentric[2] * weight_p + next_its_d.barycentric[2] * (1-weight_p);

                    // if(next_its_d_temp.isValid()){
                    //     // path2_blocked = true;
                    //     m_k *= 2;
                    // }
                    
                    if((m_x_1 > 1) || (m_x_1 < 0) || (m_x_2 > 1) || (m_x_2 < 0) || (m_x_3 < 0) || (m_x_3 > 1)){
                        mesh_hit_preserved = false;
                    }
                }
                //weight_p = 1.0;
                //weight_p_prime = 0.0;
                //jacobian_m = 1.0;

                // rectangle
                
                // Triangle
                // Point p0 = its.p0;
                // Point p1 = its.p1;
                // Point p2 = its.p2;

                // Point p0 = its.p0;
                // Point p1 = its.p1;
                // Point p2 = its.p2;
                
                // Point pg = (p0 + p1 + p2) / 3.0;
                // Triangle::rayIntersect(p0, p1, pg);
                if(!path2_blocked){
                    if(!(bRec2_d.sampledType & BSDF::EDelta)){
                    //if(true){
                        next_its_m.p = weight_p * next_its_p.p + (1 - weight_p) * next_its_d.p;
                        next_its_m.t = (next_its_m.p - its2.p).length();
                        //next_its_m = next_its_p;
                        const Vector wo2_m = normalize(next_its_m.p - its2.p);
                        ray2 = Ray(its2.p, wo2_m, ray2.time);
                        mesh_hit_preserved = false;
                        if(mesh_hit_preserved){
                            ray2.maxt = next_its_m.t - 1e-2;
                            ray2.mint = Epsilon;
                            
                            Intersection next_its_m_temp;
                            path2_blocked |= scene->rayIntersect(ray2, next_its_m_temp);
                            if(!path2_blocked){
                                BSDFSamplingRecord bRec2_m(its2, its2.toLocal(wo2_m), ERadiance);
                                bsdfVal2 = bsdf2->eval(bRec2_m);
                                bsdfPdf2 = bsdf2->pdf(bRec2_m);
                            }
                        } else {
                            
                            Intersection next_its_m_temp;
                            scene->rayIntersect(ray2, next_its_m_temp);
                            if(check_consistency(its, next_its_m_temp)){
                            //if(next_its_m_temp.isValid()){
                                BSDFSamplingRecord bRec2_m(its2, its2.toLocal(wo2_m), ERadiance);
                                bsdfVal2 = bsdf2->eval(bRec2_m);
                                bsdfPdf2 = bsdf2->pdf(bRec2_m);
                                

                            } else {
                                path2_blocked = true;
                            }
                            next_its_m = next_its_m_temp;
                        }
                        
                        // next_its_m = next_its_m_temp;

                    } else {
                        next_its_m = next_its_d;
                        const Vector wo2_m = normalize(next_its_m.p - its2.p);
                        ray2 = Ray(its2.p, wo2_m, ray2.time);
                        bsdfVal2 = bsdfWeight_d;
                        bsdfPdf2 = 1.0;
                    }
                    if(!path2_blocked){
                        G2 = std::abs(dot(next_its_m.shFrame.n, ray2.d)) / (next_its_m.t * next_its_m.t);
                    }
                    jacobian_m = G / G2;


                    path_length2 += next_its_m.t;
                    its2 = next_its_m;  
                    
                } else {
                    G2 = 1.0;
                }
                
                
                // path_pdf_1_as_1_nee = path_pdf_1_as_1 * lumPdf * G;
                path_pdf_1_as_2_nee = path_pdf_1_as_2 * lumPdf2 * G2 * jacobian_m;

                path_pdf_2_as_1_nee = path_pdf_2_as_1 * lumPdf2 * G2;
                path_pdf_2_as_2_nee = path_pdf_2_as_2 * lumPdf * G / jacobian_m;

                // path_pdf_1_as_1 = path_pdf_1_as_1 * bsdfPdf * G; 
                path_pdf_1_as_2_bsdf = path_pdf_1_as_2 * bsdfPdf2 * G2 * jacobian_m;
                
                path_pdf_2_as_1_bsdf = path_pdf_2_as_1 * bsdfPdf2 * G2;
                path_pdf_2_as_2_bsdf = path_pdf_2_as_2 * bsdfPdf * G / jacobian_m;
            }

            ////////////////////////////////////
            // path 2
            ////////////////////////////////////
            
            if(!path2_blocked && !useMixing){
                /////////////////////////////////////////////////////////
                // (0) Calculate correlation strategy probability
                /////////////////////////////////////////////////////////
                Float area = roughness * reflected_dist * reflected_dist / (reflected_cos_o);
                Float position_correlation_probability = 6.0f * area;
                position_correlation_probability = std::min(position_correlation_probability, 1.0);

                bool force_position = (strcmp(m_antithetic_sampling_mode.c_str(), "position") == 0);
                bool force_direction = (strcmp(m_antithetic_sampling_mode.c_str(), "direction") == 0);

                if(force_direction){
                    position_correlation_probability = 0;
                } else if (force_position){
                    position_correlation_probability = 1;
                }

                ////////////////////////////////////
                // (1) Position Correlation
                ////////////////////////////////////

                if(rRec.nextSample1D() < position_correlation_probability){
                    Intersection next_its = its;
                    next_its.adjustTime(ray2.time);
                    next_its.t = (next_its.p - its2.p).length();
                    const Vector wo2 = normalize(next_its.p - its2.p);
                    ray2 = Ray(its2.p, wo2, ray2.time);
                    ray2.maxt = next_its.t - 1e-2;
                    ray2.mint = Epsilon;

                    // Check discontinuity
                    Intersection its_temp;
                    path2_blocked |= scene->rayIntersect(ray2, its_temp);

                    if(!path2_blocked){
                        BSDFSamplingRecord bRec2(its2, its2.toLocal(wo2), ERadiance);
                        bsdfVal2 = bsdf2->eval(bRec2);
                        bsdfPdf2 = bsdf2->pdf(bRec2);
                        its2 = next_its;
                        if(!(bRec2.sampledType & BSDF::EDelta)){
                            G2 = std::abs(dot(its2.shFrame.n, ray2.d)) / (its2.t * its2.t);
                        } else {
                            G2 = 1.0;
                        }
                        path_length2 += its2.t;

                        if (its2.isEmitter()) {
                            value2 = its2.Le(-ray2.d);
                            dRec2.setQuery(ray2, its2);
                            hitEmitter2 = true;
                        }

                        /* If a luminaire was hit, estimate the local illumination and
                        weight using the power heuristic */
                        if (hitEmitter2) {
                            lumPdf2 = (!(bRec2.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec2) : 0;
                            modulation_weight2 = evalModulationWeight(ray2.time, path_length2);
                        }
                    }

                    // path_pdf_1_as_1_nee = path_pdf_1_as_1 * lumPdf * G;
                    path_pdf_1_as_2_nee = path_pdf_1_as_2 * lumPdf2 * G2;

                    path_pdf_2_as_1_nee = path_pdf_2_as_1 * lumPdf2 * G2;
                    path_pdf_2_as_2_nee = path_pdf_2_as_2 * lumPdf * G;

                    // path_pdf_1_as_1 = path_pdf_1_as_1 * bsdfPdf * G; 
                    path_pdf_1_as_2_bsdf = path_pdf_1_as_2 * bsdfPdf2 * G2;
                    
                    path_pdf_2_as_1_bsdf = path_pdf_2_as_1 * bsdfPdf2 * G2;
                    path_pdf_2_as_2_bsdf = path_pdf_2_as_2 * bsdfPdf * G;
                }

                ////////////////////////////////////
                // (2) Sampler Correlation
                ////////////////////////////////////
                else {
                    BSDFSamplingRecord bRec2(its2, rRec2.sampler, ERadiance);
                    Spectrum bsdfWeight2 = bsdf2->sample(bRec2, bsdfPdf2, usedSample);
                    bsdfVal2 = bsdfWeight2 * bsdfPdf2;
                    const Vector wo2 = its2.toWorld(bRec2.wo);
                    ray2 = Ray(its2.p, wo2, ray2.time);
                    scene->rayIntersect(ray2, its2);

                    // Check discontinuity
                    path2_blocked |= (!check_consistency(its, its2)); 
                    
                    if(!path2_blocked){
                        if(!(bRec2.sampledType & BSDF::EDelta)){
                            G2 = std::abs(dot(its2.shFrame.n, ray2.d)) / (its2.t * its2.t);
                        } else {
                            G2 = 1.0f;
                        }
                        path_length2 += its2.t;

                        if (its2.isEmitter()) {
                            value2 = its2.Le(-ray2.d);
                            dRec2.setQuery(ray2, its2);
                            hitEmitter2 = true;
                        }

                        /* If a luminaire was hit, estimate the local illumination and
                        weight using the power heuristic */
                        if (hitEmitter2) {
                            lumPdf2 = (!(bRec2.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec2) : 0;
                            modulation_weight2 = evalModulationWeight(ray2.time, path_length2);
                        }
                    } else {
                        bsdfVal2 *= 0.0f;
                        bsdfPdf2 *= 0.0f;
                    }

                    // path_pdf_1_as_1_nee = path_pdf_1_as_1 * lumPdf * G;
                    path_pdf_1_as_2_nee = path_pdf_1_as_2 * lumPdf2 * G;

                    path_pdf_2_as_1_nee = path_pdf_2_as_1 * lumPdf2 * G2;
                    path_pdf_2_as_2_nee = path_pdf_2_as_2 * lumPdf * G2;

                    // path_pdf_1_as_1 = path_pdf_1_as_1 * bsdfPdf * G; 
                    path_pdf_1_as_2_bsdf = path_pdf_1_as_2 * bsdfPdf2 * G;
                    
                    path_pdf_2_as_1_bsdf = path_pdf_2_as_1 * bsdfPdf2 * G2;
                    path_pdf_2_as_2_bsdf = path_pdf_2_as_2 * bsdfPdf * G2;
                }
            }

            path_throughput *= bsdfVal * G;
            path_throughput2 *= bsdfVal2 * G2;

            // MIS for BSDF 
            if(hitEmitter){
                Float p1 = path_pdf_1_as_1_bsdf;
                Float p2 = path_pdf_1_as_2_bsdf;
                Float p3 = path_pdf_1_as_1_nee;
                Float p4 = path_pdf_1_as_2_nee;
                Li += path_throughput * value * modulation_weight / p1 * miWeight4(p1, p2, p3, p4);
            }
            if(hitEmitter2){
                Float p1 = path_pdf_2_as_2_bsdf;
                Float p2 = path_pdf_2_as_1_bsdf;
                Float p3 = path_pdf_2_as_2_nee;
                Float p4 = path_pdf_2_as_1_nee;
                Li += path_throughput2 * value2 * modulation_weight2 / p1 * miWeight4(p1, p2, p3, p4);
            }

            path_pdf_1_as_1 = path_pdf_1_as_1_bsdf;
            path_pdf_1_as_2 = path_pdf_1_as_2_bsdf;
            path_pdf_2_as_1 = path_pdf_2_as_1_bsdf;
            path_pdf_2_as_2 = path_pdf_2_as_2_bsdf;

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec2.type = RadianceQueryRecord::ERadianceNoEmission;
            
            rRec.depth++;
            rRec2.depth++;
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

            Float mis_position = miWeight(path_pdf_2_as_2, path_pdf_2_as_3);
            Float mis_direction = miWeight(path_pdf_3_as_3, path_pdf_3_as_2);

            //printf("PDF path_pdf_2_as_2 %f, path_pdf_2_as_3 %f \n", path_pdf_2_as_2, path_pdf_2_as_3);
            //printf("PDF path_pdf_3_as_3 %f, path_pdf_3_as_2 %f \n", path_pdf_3_as_3, path_pdf_3_as_2);
            
            //printf("MIS WEIGHT w_p : %f, w_d : %f, sum : %f \n", mis_position, mis_direction, mis_position + mis_direction);

            if(m_preserve_primal_mis_weight_to_one){
                Li += 0.5f * neePathThroughput1 / path_pdf_1_as_1;
            } else {
                Li += 0.5f * mis_position * neePathThroughput1 / path_pdf_1_as_1;// * miWeight3(path_pdf_1_as_1, path_pdf_1_as_2, 0);
                Li += 0.5f * mis_direction * neePathThroughput1 / path_pdf_1_as_1;// * miWeight3(path_pdf_1_as_1, path_pdf_1_as_2, 0);
            }
            
            Li += 0.5f * mis_position * neePathThroughput2 / path_pdf_2_as_2;// * miWeight3(path_pdf_2_as_2, path_pdf_2_as_1, 0);
            Li += 0.5f * mis_direction * neePathThroughput3 / path_pdf_3_as_3;// * miWeight3(path_pdf_3_as_3, path_pdf_3_as_1, 0);

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
        Float ray2_time = r.time + 0.5f * m_time;
        if(ray2_time >= m_time){
            ray2_time -= m_time;
        }
        if(!m_antithetic_sampling_by_shift){
            ray2_time = m_time - r.time;
        }
        if(strcmp(m_antithetic_sampling_mode.c_str(), "mis")==0){
            return Li_helper_mis(r, rRec, ray2_time);
        }

        if(strcmp(m_antithetic_sampling_mode.c_str(), "mixed")==0){
            return Li_helper(r, rRec, ray2_time);
        }

        if(strcmp(m_antithetic_sampling_mode.c_str(), "mixed2")==0){
            return Li_helper(r, rRec, ray2_time);
        }
        
        return Li_helper(r, rRec, ray2_time);
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        if(pdfA + pdfB == 0.0f){
            return 0.0f;
        }
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    inline Float miWeight3(Float pdfA, Float pdfB, Float pdfC) const {
        if(pdfA + pdfB + pdfC == 0.0f){
            return 0.0f;
        }
        //pdfA *= pdfA;
        //pdfB *= pdfB;
        //pdfC *= pdfC;
        //pdfD *= pdfD;
        return pdfA / (pdfA + pdfB + pdfC);
    }

    inline Float miWeight4(Float pdfA, Float pdfB, Float pdfC, Float pdfD) const {
        if(pdfA + pdfB + pdfC +pdfD == 0.0f){
            return 0.0f;
        }
        //pdfA *= pdfA;
        //pdfB *= pdfB;
        //pdfC *= pdfC;
        //pdfD *= pdfD;
        return pdfA / (pdfA + pdfB + pdfC + pdfD);
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
    
    bool m_low_frequency_component_only;
    bool m_is_objects_transformed_for_time;
    bool m_force_constant_attenuation;
    bool m_correlate_time_samples;
    bool m_antithetic_sampling_by_shift;
    bool m_preserve_primal_mis_weight_to_one;
    std::string m_antithetic_sampling_mode;
};

MTS_IMPLEMENT_CLASS_S(ToFAntitheticPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ToFAntitheticPathTracer, "ToF antithetic path tracer");
MTS_NAMESPACE_END


