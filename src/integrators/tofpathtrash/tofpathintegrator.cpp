
#include "tofpathintegrator.h"
MTS_NAMESPACE_BEGIN
ToFPathCorrelatedIntegrator::ToFPathCorrelatedIntegrator(const Properties &props)
    : MonteCarloIntegrator(props) {
    
}

/// Unserialize from a binary data stream
ToFPathCorrelatedIntegrator(Stream *stream, InstanceManager *manager)
    : MonteCarloIntegrator(stream, manager) { }

Spectrum ToFPathCorrelatedIntegrator::accumulate_next_ray_Li(PathTracePart &path1, PathTracePart &path2) const
{
    return Spectrum(0.0);
}
Spectrum ToFPathCorrelatedIntegrator::accumulate_nee_Li(PathTracePart &path1, PathTracePart &path2) const
{
    return Spectrum(0.0);
}


Spectrum ToFPathCorrelatedIntegrator::trace_two_rays(const RayDifferential &r1, const RayDifferential &r2, RadianceQueryRecord &rRec){
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

    while(rRec.depth <= m_maxDepth || m_maxDepth < 0){
        bool use_positional_correlation = false;
        if(strcmp(m_antithetic_sampling_mode.c_str(), "position") == 0){
            use_positional_correlation = true;
        }
        if(path1.its.isValid()){
            const BSDF *bsdf = path1.its.getBSDF(path1.ray);
            Float its_roughness = bsdf->getRoughness(path1.prev_its, 0);
            if(its_roughness == 0.0){
                use_positional_correlation = false;
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
            PathTracePart::update_path_pdf_positional(path1, path2);
        } else {
            PathTracePart::calculate_next_ray_pdf_sampler(path1, path2);
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



MTS_IMPLEMENT_CLASS(ToFPathCorrelatedIntegrator, true, MonteCarloIntegrator)
MTS_NAMESPACE_END
