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
#include "tof_utils.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("ToFPath tracer", "Average path length", EAverage);

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
class ToFPathTracer : public MonteCarloIntegrator {
public:
    ToFPathTracer(const Properties &props)
        : MonteCarloIntegrator(props) {
        m_needOffset = true;
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
    }

    /// Unserialize from a binary data stream
    ToFPathTracer(Stream *stream, InstanceManager *manager)
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
    
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
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

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "ToFPathTracer[" << endl
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

    
    int m_wave_function_type;
};

MTS_IMPLEMENT_CLASS_S(ToFPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ToFPathTracer, "ToF path tracer");
MTS_NAMESPACE_END
