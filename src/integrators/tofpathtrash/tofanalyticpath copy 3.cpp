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
        if(std::abs(a) < 1){
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

    Spectrum Li_from(const RayDifferential &r, RadianceQueryRecord &rRec, Float startTime, Float endTime) const {
        
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        ray.setTime(startTime);

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

        Intersection its_T;// = its;
        Ray ray_T = Ray(ray.o, ray.d, endTime);

        Intersection prev_its;
        Intersection prev_its_T;
        
        scene->rayIntersect(ray_T, its_T);
        // Point p_at_T = its_T.p;
        //if((p_at_T - its.p).length() > 1e-2){
        //    p_at_T = its.p;
        //}

        Float path_length_at_T = its_T.t;
        Float f_value_ratio = 1.0;

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
                Float dem_length_weight = evalIntegratedModulationWeight(startTime, endTime, path_length, path_length, 0);
                Li += throughput * its.Le(-ray.d) * dem_length_weight;
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
                        
                        Point em_p_at_T = dRec.p;

                        Float em_path_length = path_length + dRec.dist;
                        Float em_path_length_at_T = path_length_at_T + (em_p_at_T - its_T.p).length();

                        Float dist_sqr_1 = dRec.dist * dRec.dist;
                        Float cos_i_1 = dot(its.shFrame.n, dRec.d);
                        Float cos_o_1 = emitter->isOnSurface()?dot(dRec.n, -dRec.d):1;
                        Float f_1 = std::abs(cos_i_1) * std::abs(cos_o_1) / dist_sqr_1;

                        Float dist_sqr_2 = (em_p_at_T - its_T.p).lengthSquared();
                        Float cos_i_2 = dot(its_T.shFrame.n, normalize((em_p_at_T - its_T.p)));
                        Float cos_o_2 = emitter->isOnSurface()?dot(dRec.n, -normalize((em_p_at_T - its_T.p))):1;
                        Float f_2 = std::abs(cos_i_2) * std::abs(cos_o_2) / dist_sqr_2;

                        Float f_value_ratio_em = f_value_ratio * f_2 / f_1;
                        Float em_length_weight = evalIntegratedModulationWeight(startTime, endTime, em_path_length, em_path_length_at_T, f_value_ratio_em - 1);
                        // return Spectrum(std::abs(cos_i_1));

                        Li += throughput * value * bsdfVal * weight * em_length_weight;
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

            prev_its = its;
            prev_its_T = its_T;

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

            its_T = its;
            its_T.adjustTime(endTime);

            path_length += its.t;
            path_length_at_T += (its_T.p - prev_its_T.p).length();
            
            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            Float dist_sqr_1 = its.t * its.t;
            Float cos_i_1 = dot(prev_its.shFrame.n, ray.d);
            Float cos_o_1 = dot(its.shFrame.n, -ray.d);
            Float f_1 = std::abs(cos_i_1) * std::abs(cos_o_1) / dist_sqr_1;

            Float dist_sqr_2 = (its_T.p - prev_its_T.p).lengthSquared();
            Float cos_i_2 = dot(prev_its_T.shFrame.n, normalize((its_T.p - prev_its_T.p)));
            Float cos_o_2 = dot(its_T.shFrame.n, -normalize((its_T.p - prev_its_T.p)));
            Float f_2 = std::abs(cos_i_2) * std::abs(cos_o_2) / dist_sqr_2;

            f_value_ratio = f_value_ratio * f_2 / f_1;

            //p_at_T = its_T.p;

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;

                Float length_weight = evalIntegratedModulationWeight(startTime, endTime, path_length, path_length_at_T, f_value_ratio - 1);

                Li += throughput * value * miWeight(bsdfPdf, lumPdf) * length_weight;
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
        
        //Li = Spectrum(path_length_at_T - path_length);

        return Li;
    }

    bool check_consistency(Intersection& its1, Intersection& its2) const{
        bool consistent = its1.isValid() && its2.isValid() && (its1.p - its2.p).length() < 0.1 && dot(its1.shFrame.n, its2.shFrame.n) > 0.9;
        // bool consistent = (its1.shape == its2.shape);
        
        return consistent;
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        // return Li_from(r, rRec, 0, m_time);
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection its;
        RayDifferential ray(r);

        Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        scene->rayIntersect(ray, its);
        // rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;


        Intersection its_T;// = its;
        Ray ray_T = Ray(ray.o, ray.d, m_time);

        Intersection prev_its;
        Intersection prev_its_T;
        
        scene->rayIntersect(ray_T, its_T);
        ray_T.mint=Epsilon;
        
        //bool consistent = (its.p - its_T.p).length() < 0.1 && dot(its.shFrame.n, its_T.shFrame.n) > 0.9 && (its.primIndex == its_T.primIndex);
        // printf("This is cosistent\n");
        bool consistent = check_consistency(its, its_T);
        // check consistency
        if(!consistent)
        // if(false)
        {
            Float start = 0;
            Float end = m_time;
            Float hit_time = 0.5f * (end + start);


            for(int i=0; i < 10; i++){
                hit_time = 0.5f * (end + start);
                ray_T.setTime(hit_time);
                ray_T.mint=Epsilon;

                Intersection its_temp;
                scene->rayIntersect(ray_T, its_temp);

                //bool hit = length(its.p, its_T) < 0.1;//its_temp.shape == its.shape;//its.shape->rayIntersect(ray_T, ray_T.mint, ray_T.maxt);
                
                consistent = check_consistency(its, its_temp);
                
                if(fabsf(its.p.x - 0.5465)< 0.0001){
                    //printf("its p : %f, %f, %f \n", its.p.x, its.p.y, its.p.z);
                    //printf("its_temp p%d : %f, %f, %f \n", i, its_temp.p.x, its_temp.p.y, its_temp.p.z);
                }
                
                if(consistent){
                    start = hit_time;
                } else {
                    end = hit_time;
                }
            }
            // printf("Hit time: %f, %f\n", start, end);

            RadianceQueryRecord rRec2 = rRec;

            Spectrum Li1 = Li_from(r, rRec, 0, start);
            Spectrum Li2 = Li_from(r, rRec2,  end, m_time);

            // return Li_from(r, rRec, 0, m_time);
            
            return Li1 + Li2;
            
        } else {
            return Li_from(r, rRec, 0, m_time);
        }
        
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

    Float m_time;
};

MTS_IMPLEMENT_CLASS_S(ToFAnalyticPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ToFAnalyticPathTracer, "ToF analytic path tracer");
MTS_NAMESPACE_END
