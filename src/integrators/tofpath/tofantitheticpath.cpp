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

        // bool consistency = (dot(its2.shFrame.n, its.shFrame.n) > 0.9) && (std::abs(its2.t - its.t) < 0.1);
        path2_blocked = (!its2.isValid());// || (!consistency);
        
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

            /* Estimate the direct illumination if this is requested */
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

                        Float em_path_length = path_length + dRec.dist;

                        Float em_modulation_weight = evalModulationWeight(ray.time, em_path_length);

                        if(emitter->isOnSurface()){
                            Float G = std::abs(dot(dRec.d, dRec.n)) / (dRec.dist * dRec.dist);
                            neePathNEEPdf = path_pdf * dRec.pdf * G;
                            neePathBSDFPdf = path_pdf * bsdfPdf * G;
                            neePathThroughput = path_throughput * bsdfVal * G *  value * em_modulation_weight;
                        } else {
                            neePathNEEPdf = dRec.pdf * path_pdf;
                            neePathBSDFPdf = 0;
                            neePathThroughput = path_throughput * bsdfVal * value * em_modulation_weight;
                        }
                    }
                }

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
                            neePathNEEPdf2 = path_pdf2 * dRec2.pdf * G2;
                            neePathBSDFPdf2 = path_pdf2 * bsdfPdf2 * G2;
                            neePathThroughput2 = path_throughput2 * bsdfVal2 * G2 *  value2 * em_modulation_weight2;
                        } else {
                            neePathNEEPdf2 = dRec2.pdf * path_pdf2;
                            neePathBSDFPdf2 = 0;
                            neePathThroughput2 = path_throughput2 * bsdfVal2 * value2 * em_modulation_weight2;
                        }
                    }
                    
                }
            }

            if(neePathNEEPdf > 0){
                Li += neePathThroughput / neePathNEEPdf * miWeight4(neePathNEEPdf, neePathBSDFPdf, neePathNEEPdf2, neePathBSDFPdf2);
            }
            if(neePathNEEPdf2 > 0){
                Li += neePathThroughput2 / neePathNEEPdf2 * miWeight4(neePathNEEPdf2, neePathBSDFPdf2, neePathNEEPdf, neePathBSDFPdf);
            }

            if(rRec.depth + 1>=m_maxDepth){
                break;
            }

            
            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
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

            G = std::abs(dot(its.shFrame.n, ray.d)) / (its.t * its.t);
            path_length += its.t;

            Float pathBSDFPdf = 0.0f;
            Float pathNEEPdf = 0.0f;
            Spectrum pathThroughput = Spectrum(0.0f);
            Float pathBSDFPdf2 = 0.0f;
            Float pathNEEPdf2 = 0.0f;
            Spectrum pathThroughput2 = Spectrum(0.0f);


            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;

                Float modulation_weight = evalModulationWeight(ray.time, path_length);

                if(!(bRec.sampledType & BSDF::EDelta)){
                    pathNEEPdf = path_pdf * lumPdf * G;
                    pathBSDFPdf = path_pdf * bsdfPdf * G;
                    pathThroughput=path_throughput * bsdfVal * value * G * modulation_weight;
                } else {
                    pathNEEPdf=0;
                    pathBSDFPdf=path_pdf;
                    pathThroughput=path_throughput * bsdfVal * value * modulation_weight;
                }
            }

            if(!(bRec.sampledType & BSDF::EDelta)){
                path_pdf *= bsdfPdf * G;
                path_throughput *= bsdfVal * G;
            } else {
                path_pdf *= bsdfPdf;
                path_throughput *= bsdfVal;
            }

            if(!path2_blocked){
                Intersection next_its = its;
                next_its.adjustTime(ray2.time);
                next_its.t = (next_its.p - its2.p).length();
                const Vector wo2 = normalize(next_its.p - its2.p);

                ray2 = Ray(its2.p, wo2, ray2.time);
                ray2.maxt = next_its.t - 1e-2;
                ray2.mint = Epsilon;
                Intersection its_temp;
                path2_blocked |= scene->rayIntersect(ray2, its_temp);

                if(!path2_blocked){
                    BSDFSamplingRecord bRec2(its2, its2.toLocal(wo2), ERadiance);
                    Spectrum bsdfVal2 = bsdf2->eval(bRec2);
                    Float bsdfPdf2 = bsdf2->pdf(bRec2);

                    its2 = next_its;
                    
                    Spectrum value2;
                    if (its2.isEmitter()) {
                        value2 = its2.Le(-ray2.d);
                        dRec2.setQuery(ray2, its2);
                    }

                    G2 = std::abs(dot(its2.shFrame.n, ray2.d)) / (its2.t * its2.t);
                    path_length2 += its2.t;

                    /* If a luminaire was hit, estimate the local illumination and
                    weight using the power heuristic */
                    if (hitEmitter &&
                        (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                        const Float lumPdf2 = (!(bRec2.sampledType & BSDF::EDelta)) ?
                            scene->pdfEmitterDirect(dRec2) : 0;

                        Float modulation_weight2 = evalModulationWeight(ray2.time, path_length2);

                        if(!(bRec2.sampledType & BSDF::EDelta)){
                            pathNEEPdf2 = path_pdf2 * lumPdf2 * G2;
                            pathBSDFPdf2 = path_pdf2 * bsdfPdf2 * G2;
                            pathThroughput2 = path_throughput2 * bsdfVal2 * value2 * G2 * modulation_weight2;
                        } else {
                            pathNEEPdf2=0;
                            pathBSDFPdf2=path_pdf2;
                            pathThroughput2=path_throughput2 * bsdfVal2 * value2 * modulation_weight2;
                        }
                    }

                    if(!(bRec2.sampledType & BSDF::EDelta)){
                        path_pdf2 *= bsdfPdf2 * G2;
                        path_throughput2 *= bsdfVal2 * G2;
                    } else {
                        path_pdf2 *= bsdfPdf2;
                        path_throughput2 *= bsdfVal2;
                    }
                }
            }
            
            if(pathBSDFPdf > 0){
                Li += pathThroughput / pathBSDFPdf * miWeight4(pathBSDFPdf, pathNEEPdf, pathBSDFPdf2, pathNEEPdf2);
            }
            if(pathBSDFPdf2 > 0){
                Li += pathThroughput2 / pathBSDFPdf2 * miWeight4(pathBSDFPdf2, pathNEEPdf2, pathBSDFPdf, pathNEEPdf);
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;
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
        
        return Li_helper(r, rRec, ray2_time);
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    inline Float miWeight4(Float pdfA, Float pdfB, Float pdfC, Float pdfD) const {
        //pdfA *= pdfA;
        //pdfB *= pdfB;
        //pdfC *= pdfC;
        //pdfD *= pdfD;

        // printf("%f, %f, %f, %f\n", pdfA, pdfB, pdfC, pdfD);

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
};

MTS_IMPLEMENT_CLASS_S(ToFAntitheticPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ToFAntitheticPathTracer, "ToF antithetic path tracer");
MTS_NAMESPACE_END
