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

    void Li_first_path(const RayDifferential &r, RadianceQueryRecord &rRec, 
        std::vector<Intersection> &path, 
        std::vector<DirectSamplingRecord> &neePath,
        std::vector<Spectrum> &neeValue,
        std::vector<Spectrum> &pathThroughputs,
        std::vector<Float> &pathPdfs) const {
            
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        // Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        //Spectrum throughput(1.0f);
        Float eta = 1.0f;

        Float path_length = 0;
        path_length += its.t;

        path.push_back(Intersection(its));

        Float path_pdf = 1.0f;
        Spectrum path_throughput(1.0f);

        if (its.isValid()) {
            Float G = std::abs(dot(its.shFrame.n, ray.d)) / (its.t * its.t);
            path_pdf *= G;
            path_throughput *= G;
        }
        
        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    // Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
            {
                Float modulation_weight = evalModulationWeight(ray.time, path_length);
                // Li += modulation_weight * throughput * its.Le(-ray.d);
            }

            /* Include radiance from a subsurface scattering model if requested */
            // if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
            //    Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

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
                neeValue.push_back(value);
                neePath.push_back(dRec);
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

                        //Spectrum Li_em = throughput * value * bsdfVal * weight * em_modulation_weight;
                        // Li += Li_em;

                        // path_throughput *= bsdfVal * value * em_modulation_weight;

                        // Float em_path_pdf = bsdfPdf * std::abs(its.shFrame.n, ray.d) / (its.t * its.t);
                        // Float em_path_throughput *= bsdfWeight * bsdfPdf * std::abs(its.shFrame.n, ray.d) / (its.t * its.t);

                        pathPdfs.push_back(path_pdf);
                        pathThroughputs.push_back(path_throughput * bsdfVal * value * em_modulation_weight);
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

            path_pdf *= bsdfPdf * std::abs(dot(its.shFrame.n, ray.d)) / (its.t * its.t);
            path_throughput *= bsdfWeight * bsdfPdf * std::abs(dot(its.shFrame.n, ray.d)) / (its.t * its.t);

            path.push_back(Intersection(its));

            path_length += its.t;
            /* Keep track of the throughput and relative
               refractive index along the path */
            // throughput *= bsdfWeight;
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
                
                // Li += throughput * value * miWeight(bsdfPdf, lumPdf) * modulation_weight;
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

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;
        // Li = Spectrum(-1.0);

        // return Li;
        return;
    }

    void Li_second_path(const RayDifferential &r, RadianceQueryRecord &rRec, 
        const std::vector<Intersection> path, 
        const std::vector<DirectSamplingRecord> neePath,
        const std::vector<Spectrum> neeValue,
        std::vector<Spectrum> &pathThroughputs,
        std::vector<Float> &pathPdfs) const {

        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        // Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        // Spectrum throughput(1.0f);
        Float eta = 1.0f;

        Float path_length = 0;
        path_length += its.t;

        Float path_pdf = 1.0f;
        Spectrum path_throughput(1.0f);

        if (its.isValid()) {
            Float G = std::abs(dot(its.shFrame.n, ray.d)) / (its.t * its.t);
            path_pdf *= G;
            path_throughput *= G;
        }

        if(!path.at(0).isValid()){
            return;
        }

        while (rRec.depth <= path.size() || m_maxDepth < 0) {
            if (!its.isValid()) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    // Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
            {
                Float modulation_weight = evalModulationWeight(ray.time, path_length);
                // Li += modulation_weight * throughput * its.Le(-ray.d);
            }

            /* Include radiance from a subsurface scattering model if requested */
            // if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
            //    Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
                    * Frame::cosTheta(its.wi) >= 0)) {

                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }

            // return std::make_pair(true, Li);

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            if(neePath.size() < rRec.depth){
                break;
            }
            //DirectSamplingRecord dRec = neePath.at(rRec.depth - 1);//(its);
            //dRec.d = normalize(dRec.p - its.p);
            //Float dRecdistSquared = (dRec.p - its.p).lengthSquared();
            // printf("ADDED!!\n");
            DirectSamplingRecord dRec(its);

            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                // Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
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

                        Float em_path_length = path_length + dRec.dist;//(dRec.p - its.p).length();

                        Float em_modulation_weight = evalModulationWeight(ray.time, em_path_length);
                        
                        //Spectrum Li_em = throughput * value * bsdfVal * weight * em_modulation_weight;
                        //Li += Li_em;

                        // path_throughput *= bsdfVal * value * em_modulation_weight;

                        pathPdfs.push_back(path_pdf);
                        pathThroughputs.push_back(path_throughput * bsdfVal * value * em_modulation_weight);
                    }
                }
            }

            // return std::make_pair(true, Li);

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            if(rRec.depth >= path.size()){
                break;
            }
            Intersection next_its = path.at(rRec.depth);
            
            if(!next_its.isValid()){
                return;
            }
            next_its.adjustTime(ray.time);

            const Vector wo = normalize(next_its.p - its.p);
            Float ray_length = (next_its.p - its.p).length();

            /* Allocate a record for querying the BSDF */
            BSDFSamplingRecord bRec(its, its.toLocal(wo), ERadiance);

            /* Evaluate BSDF * cos(theta) */
            Spectrum bsdfVal = bsdf->eval(bRec);
            Float bsdfPdf = bsdf->pdf(bRec);
            // Spectrum bsdfWeight = bsdfVal / bsdfPdf * next_its.t * next_its.t / (ray_length * ray_length);

            path_pdf *= bsdfPdf * std::abs(dot(next_its.shFrame.n, wo)) / (ray_length * ray_length);
            path_throughput *= bsdfVal * std::abs(dot(next_its.shFrame.n, wo)) / (ray_length * ray_length);
            
            /* Sample BSDF * cos(theta) */
            // Float bsdfPdf;
            // BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            // Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            // if (bsdfWeight.isZero())
            //     break;

            // /* Sample BSDF * cos(theta) */
            // Float bsdfPdf;
            // BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            // Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            // if (bsdfWeight.isZero())
            //     break;

            // scattered |= bRec.sampledType != BSDF::ENull;

            // /* Prevent light leaks due to the use of shading normals */
            // const Vector wo = its.toWorld(bRec.wo);
            // Float woDotGeoN = dot(its.geoFrame.n, wo);
            // if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
            //     break;

            bool hitEmitter = false;
            Spectrum value;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);
            ray.maxt = ray_length - 1e-2;
            ray.mint = Epsilon;
            if(scene->rayIntersect(ray, its)){
                return;
            }

            its = next_its;

            if (its.isEmitter()) {
                value = its.Le(-ray.d);
                dRec.setQuery(ray, its);
                hitEmitter = true;
            }

            path_length += ray_length;
            /* Keep track of the throughput and relative
               refractive index along the path */
            // throughput *= bsdfWeight;//bsdfVal / bsdfPdf;
            eta *= bRec.eta;

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                const Float lumPdf = scene->pdfEmitterDirect(dRec);

                Float modulation_weight = evalModulationWeight(ray.time, path_length);
                
                //Li += throughput * value * miWeight(bsdfPdf, lumPdf) * modulation_weight;
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

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;
        // Li = Spectrum(-1.0);
        
        return;
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        std::vector<Intersection> path;
        std::vector<DirectSamplingRecord> neePath;
        std::vector<Spectrum> neeValue;
        std::vector<Spectrum> pathThroughputs;
        std::vector<Float> pathPdfs;

        RadianceQueryRecord rRec2 = rRec;
        RayDifferential r2 = RayDifferential(r);

        r2.time = r.time + 0.5f * m_time;
        if(r2.time >= m_time){
            r2.time -= m_time;
        }
        
        // if(m_illumination_modulation_frequency_mhz == m_sensor_modulation_frequency_mhz){
        //     r2.time = m_time - r.time;
        // } else {
        // }
        // r2.time = m_time - r.time;

        Li_first_path(r, rRec, path, neePath, neeValue, pathThroughputs, pathPdfs);
        int path_n = pathThroughputs.size();
        for(int i=0; i<path_n; i++){
            // printf("Path %d, PDF: %f\n", i, pathPdfs.at(i));
        }

        std::vector<Spectrum> pathThroughputs2;
        std::vector<Float> pathPdfs2;
        Li_second_path(r2, rRec2, path, neePath, neeValue, pathThroughputs2, pathPdfs2);

        Spectrum Li(0.0f);
        for(int i=0; i<path_n; i++){
            Spectrum f1 = pathThroughputs.at(i);
            Float p1 = pathPdfs.at(i);
            if(i + 1 <= (pathThroughputs2.size())){
                Spectrum f2 = pathThroughputs2.at(i);
                Float p2 = pathPdfs2.at(i);
                Li += (f1 + f2) / (p1 + p2);
                // Li += 0.5 * (f1 / p1 + f2 / p2);
            } else 
            {
                Li += f1 / p1;
            }
        }

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
