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
#include "doppler_tof.h"


MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Doppler ToF path tracer", "Average path length", EAverage);

class DopplerToFPathTracer : public MonteCarloIntegrator, public DopplerToF {
public:
    DopplerToFPathTracer(const Properties &props)
        : MonteCarloIntegrator(props), DopplerToF(props) {
        m_offset = props.getFloat("image_offset", 1.0); // output can be negative, so add offset to make it positive

        m_time_sampling_mode = props.getString("time_sampling_mode", "uniform");
        m_use_full_time_stratification = props.getBoolean("use_full_time_stratification", false);
        m_spatial_correlation_mode = props.getString("spatial_correlation_mode", "none");

        // Use time antithetic samples?
        if(strcmp(m_time_sampling_mode.c_str(), "antithetic_mirror") == 0){
            std::string antithetic_shift_string = props.getString("antithetic_shifts", "0.0");
            m_antithetic_shifts = parse_float_array_from_string(antithetic_shift_string);
        } else {
            int antithetic_shift_number = props.getInteger("antithetic_shifts_number", 0);
            if(antithetic_shift_number > 0){
                m_antithetic_shifts.clear();
                for(int i=1; i<antithetic_shift_number; i++){
                    float t = 1.0 / ((float) antithetic_shift_number) * i;
                    m_antithetic_shifts.push_back(t);
                    std::cout << t << std::endl;
                }
            } else {
                std::string antithetic_shift_string = props.getString("antithetic_shifts", "0.5");
                m_antithetic_shifts = parse_float_array_from_string(antithetic_shift_string);
            }
        }
    }

    /// Unserialize from a binary data stream
    DopplerToFPathTracer(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    
    // Dummy Li
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        Spectrum Li(0.0f);
        return Li;
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
        path_length += its.t * eta;

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

            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;
            path_length += its.t * eta;

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

        bool analytic = strcmp(m_time_sampling_mode.c_str(), "analytic") == 0;

        while(rRec.depth <= m_maxDepth || m_maxDepth < 0){
            bool use_positional_correlation = false;
            if(strcmp(m_spatial_correlation_mode.c_str(), "ray_position") == 0){
                use_positional_correlation = true;
            }
            if(strcmp(m_spatial_correlation_mode.c_str(), "ray_selective") == 0 && 
                path1.its.isValid() && path2.its.isValid()){
                const BSDF *bsdf1 = path1.its.getBSDF(path1.ray);
                Float its1_roughness = bsdf1->getRoughness(path1.its, 0);
                const BSDF *bsdf2 = path1.its.getBSDF(path2.ray);
                Float its2_roughness = bsdf2->getRoughness(path2.its, 0);
                its1_roughness = std::min(its1_roughness, 2.0);
                its2_roughness = std::min(its2_roughness, 2.0);

                if(its1_roughness > 0.5 && its2_roughness > 0.5){
                    use_positional_correlation = true;
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
                if(analytic){
                    Li += accumulate_next_ray_Li_analytic(path1, path2, 1);
                } else {
                    Li += accumulate_next_ray_Li(path1, path2);
                    Li += accumulate_next_ray_Li(path2, path1);
                }
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
            if(analytic){
                Float jacobian = path2.path_pdf_as(path2) == 0.0 ? 0.0 : path1.path_pdf_as(path1) / path2.path_pdf_as(path2);
                Li += accumulate_nee_Li_analytic(path1, path2, jacobian * path2.G_nee / path1.G_nee);
            } else {
                Li += accumulate_nee_Li(path1, path2);
                Li += accumulate_nee_Li(path2, path1);
            }


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
            if(analytic){
                Float jacobian = path2.path_pdf_as(path2) == 0.0 ? 0.0 : path1.path_pdf_as(path1) / path2.path_pdf_as(path2);
                Li += accumulate_next_ray_Li_analytic(path1, path2, jacobian);
            } else { 
                Li += accumulate_next_ray_Li(path1, path2);
                Li += accumulate_next_ray_Li(path2, path1);
            }

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            rRec2.type = RadianceQueryRecord::ERadianceNoEmission;
            
            rRec.depth++;
            rRec2.depth++;
        }
        return Li;
    }

    // (Experimental) Trace ray with two paired samples + two MIS pairs (total 5)
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
        RayDifferential ray4(r1);
        rRec4.rayIntersect(ray4);
        PathTracePart path4(ray4, rRec4, 3);

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

            if(path4.path_terminated){
                path3.set_path_pdf_as(path2, 0.0);
                path4.set_path_pdf_as(path4, 0.0);
            }

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
                path4.get_next_ray_from_sample(path2.sample);
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
            
        }
        return Li;
    }

    void renderBlock(const Scene *scene,
        const Sensor *sensor, Sampler *sampler, ImageBlock *block,
        const bool &stop, const std::vector< TPoint2<uint8_t> > &points) const {

        Float diffScaleFactor = 1.0f /
            std::sqrt((Float) sampler->getSampleCount());

        bool needsApertureSample = sensor->needsApertureSample();
        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;

        block->clear();

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) /* Don't compute an alpha channel if we don't have to */
            queryType &= ~RadianceQueryRecord::EOpacity;

        int n_time_samples = this->m_antithetic_shifts.size() + 1;
        std::vector<Float> sampled_times(n_time_samples);

        for (size_t i = 0; i<points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            if (stop)
                break;
            
            sampler->generate(offset);
            
            // Need permutation?
            std::vector <Float *> sampleArrays1D;
            if((strcmp(m_time_sampling_mode.c_str(), "stratified") == 0) && m_use_full_time_stratification){
                for(size_t i=0; i<n_time_samples; i++){
                    sampleArrays1D.push_back(new Float[sampler->getSampleCount()]);
                    uint64_t seed = scene->getFilm()->getSize().x * offset.y + offset.x;
                    seed = seed * n_time_samples + i;
                    ref<Random> random = new Random(seed);
                    latinHypercube(random, sampleArrays1D[i], sampler->getSampleCount(), 1);
                }
            }

            for (size_t j = 0; j<sampler->getSampleCount(); j++) {
                rRec.newQuery(queryType, sensor->getMedium());

                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));            
                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                timeSample = rRec.nextSample1D();

                // 1. Time sampling
                Float t0 = timeSample;

                // (1) uniform sampling
                if(strcmp(m_time_sampling_mode.c_str(), "uniform") == 0){
                    sampled_times[0] = t0;
                    for(int i=1; i<n_time_samples; i++){
                        sampled_times[i] = rRec.nextSample1D();
                    }
                }

                // (2) stratified sampling
                else if(strcmp(m_time_sampling_mode.c_str(), "stratified") == 0){
                    if(m_use_full_time_stratification){
                        for(int i=0; i<n_time_samples; i++){
                            sampled_times[i] = (sampleArrays1D[i][j] + i) * 1.0 / n_time_samples;
                        }
                    } else {
                        for(int i=0; i<n_time_samples; i++){
                            sampled_times[i] = (rRec.nextSample1D() + i) * 1.0 / n_time_samples;
                        }
                    }
                }

                // (3) antithetic sampling
                else if(strcmp(m_time_sampling_mode.c_str(), "antithetic") == 0){
                    if(m_use_full_time_stratification){
                        t0 = (Float(j) + t0) / Float(sampler->getSampleCount());
                    }
                    sampled_times[0] = t0;
                    for(int i=1; i<n_time_samples; i++){
                        sampled_times[i] = std::fmod(t0 + m_antithetic_shifts.at(i-1), 1.0);
                    }
                }

                // (4) antithetic sampling mirror
                else if(strcmp(m_time_sampling_mode.c_str(), "antithetic_mirror") == 0){
                    if(m_use_full_time_stratification){
                        t0 = (Float(j) + t0) / Float(sampler->getSampleCount());
                    }
                    sampled_times[0] = t0;
                    sampled_times[1] = std::fmod(1.0 - t0 + m_antithetic_shifts.at(0), 1.0);
                }

                // (5) analytic
                else if(strcmp(m_time_sampling_mode.c_str(), "analytic") == 0){
                    sampled_times[0] = 0.0;
                    sampled_times[1] = 1.0;
                }

                // 2. Spatial correlation
                Spectrum spec(0.0f);
                Spectrum spec_offset(m_offset);
                Float time_weight = (strcmp(m_time_sampling_mode.c_str(), "analytic") == 0) ? 1.0 : m_time;
                
                // (1) no correlation
                if(strcmp(m_spatial_correlation_mode.c_str(), "none") == 0){
                    for(int i=0; i<n_time_samples; i++){
                        RayDifferential sensorRay_i;
                        RadianceQueryRecord rRec_i = rRec;

                        Point2 samplePos_i = (i==0) ? samplePos : Point2(offset) + Vector2(rRec_i.nextSample2D());
                        Point2 apertureSample_i = (i==0 || !needsApertureSample) ? apertureSample : rRec_i.nextSample2D();
                        
                        Spectrum spec_i = sensor->sampleRayDifferential(
                            sensorRay_i, samplePos_i, apertureSample_i, sampled_times.at(i));
                        sensorRay_i.scaleDifferential(diffScaleFactor);
                        spec_i *= Li_with_single_sample(sensorRay_i, rRec_i);
                        block->put(samplePos_i, spec_offset + spec_i * time_weight, rRec_i.alpha);
                        sampler->advance();
                    }
                }

                // (2) pixel correlation
                else if(strcmp(m_spatial_correlation_mode.c_str(), "pixel") == 0){
                    for(int i=0; i<n_time_samples; i++){
                        RayDifferential sensorRay_i;
                        RadianceQueryRecord rRec_i = rRec;

                        Spectrum spec_i = sensor->sampleRayDifferential(
                            sensorRay_i, samplePos, apertureSample, sampled_times.at(i));
                        sensorRay_i.scaleDifferential(diffScaleFactor);
                        spec += spec_i * Li_with_single_sample(sensorRay_i, rRec_i);
                        sampler->advance();
                    }
                    spec /= n_time_samples;
                    block->put(samplePos, spec_offset + spec * time_weight, rRec.alpha);
                }

                // (3) sampler correlation
                else if(strcmp(m_spatial_correlation_mode.c_str(), "sampler") == 0){
                    sampler->saveState();

                    for(int i=0; i<n_time_samples; i++){
                        RayDifferential sensorRay_i;
                        RadianceQueryRecord rRec_i = rRec;
                        sampler->loadSavedState();

                        Spectrum spec_i = sensor->sampleRayDifferential(
                            sensorRay_i, samplePos, apertureSample, sampled_times.at(i));
                        sensorRay_i.scaleDifferential(diffScaleFactor);
                        spec += spec_i * Li_with_single_sample(sensorRay_i, rRec_i);
                    }
                    for(int i=0; i<n_time_samples; i++){
                        sampler->advance();
                    }
                    spec /= n_time_samples;
                    block->put(samplePos, spec_offset + spec * time_weight, rRec.alpha);
                }

                // (4) per-ray correlation (sampler / position / mis / stochastic)
                else {
                    RayDifferential r1;
                    Spectrum spec_1 = sensor->sampleRayDifferential(
                        r1, samplePos, apertureSample, sampled_times.at(0));
                    r1.scaleDifferential(diffScaleFactor);

                    RayDifferential r2;
                    Spectrum spec_2 = sensor->sampleRayDifferential(
                        r2, samplePos, apertureSample, sampled_times.at(1));
                    r2.scaleDifferential(diffScaleFactor);

                    spec = spec_1;  // spec_1 == spec_2

                    if(strcmp(m_spatial_correlation_mode.c_str(), "mis") == 0){
                        spec = spec_1 * Li_with_MIS(r1, r2, rRec);  
                    } else {
                        spec = spec_1 * Li_with_paired_sample(r1, r2, rRec);
                    }
                        
                    block->put(samplePos, spec_offset + spec * time_weight, rRec.alpha);
                    sampler->advance();
                } 
            }
        }
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "DopplerToFPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    std::string m_time_sampling_mode;
    std::string m_spatial_correlation_mode;

    Float m_antithetic_shift;
    std::vector<float> m_antithetic_shifts;
    bool m_use_full_time_stratification;
};

MTS_IMPLEMENT_CLASS_S(DopplerToFPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(DopplerToFPathTracer, "Doppler ToF path tracer");
MTS_NAMESPACE_END


