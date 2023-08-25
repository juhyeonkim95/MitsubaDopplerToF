#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include "tof_utils.h"


using namespace mitsuba;

class PathTracePart
{
public:
    PathTracePart() = default;

    PathTracePart(Ray& ray, RadianceQueryRecord& rRec, int index) 
        : ray(ray), rRec(rRec), index(index), its(rRec.its){
        // rRec.rayIntersect(ray);
        hitEmitter = false;
        hitEmitterValue = Spectrum(0.0);
        G = 1;
        if (its.isValid() && its.isEmitter()) {
            hitEmitterValue = its.Le(-ray.d);
            hitEmitter = true;
        }

        ray.mint = Epsilon;
        path_length = its.t;
        em_path_length = 0.0f;
        m_path_pdf_as[0] = 1.0f;
        m_path_pdf_as[1] = 1.0f;
        m_path_pdf_as[2] = 1.0f;
        m_path_pdf_as[3] = 1.0f;
        m_path_pdf_as[4] = 1.0f;
        //m_path_pdf_as_nee[0] = 0.0f;
        //m_path_pdf_as_nee[1] = 0.0f;
        //m_path_pdf_as_nee[2] = 0.0f;

        path_pdf_next_ray_as_next_ray = 1.0;
        path_pdf_next_ray_as_nee = 0.0;
        
        //path_pdf = 1.0f;
        //path_pdf_as_other = 1.0f;
        path_throughput = Spectrum(1.0);
        scene = rRec.scene;
        eta = 1.0;
        scattered = false;
        path_terminated = !its.isValid();
    }

     PathTracePart& operator=(const PathTracePart& copy) = default;

    inline void set_path_pdf_as(PathTracePart& other, Float pdf){
        m_path_pdf_as[other.index] *= pdf;
    }

    // inline void set_path_pdf_as_nee(PathTracePart& other, Float pdf){
    //     m_path_pdf_as_nee[other.index] *= pdf;
    // }

    inline Float path_pdf_as(PathTracePart& other){
        return m_path_pdf_as[other.index];
    }

    // inline Float path_pdf_as_nee(PathTracePart& other){
    //     return m_path_pdf_as_nee[other.index];
    // }


    int index;
    //Float path_pdf;
    //Float path_pdf_as_other;

    Float m_path_pdf_as[5];
    //Float m_path_pdf_as_nee[3];
    
    //Float[3] path_pdf_as_nee;
    //Float[3] path_pdf_as_bsdf;

    Float bsdfPdf;
    Float lumPdf;

    Float path_pdf_nee_as_next_ray;
    Float path_pdf_nee_as_nee;
    Float path_pdf_next_ray_as_next_ray;
    Float path_pdf_next_ray_as_nee;

    Float mis_weight_nee;
    Spectrum path_throughput_nee;
    Spectrum path_throughput;
    RadianceQueryRecord rRec;
    Intersection its;
    Intersection next_its;
    Intersection prev_its;
    DirectSamplingRecord dRec;
    Float path_length;
    Float em_path_length;
    RayDifferential ray;
    bool m_strictNormals;
    bool m_hideEmitters;
    bool path_terminated;
    bool scattered;
    const Scene *scene;
    Float eta;
    Float G;
    Float G_nee;
    
    bool hitEmitter;
    Spectrum hitEmitterValue;
    Spectrum neeEmitterValue;

    Spectrum bsdfVal;
    Vector wo;
    Point2 sample;

    void nee_trace(Point2 &sample, bool testVisibility=true){
        bsdfPdf = 0.0f;
        lumPdf = 0.0f;
        path_throughput_nee = Spectrum(0.0f);
        G_nee = 1.0;
        if(this->path_terminated)
            return;

        const BSDF *bsdf = its.getBSDF(ray);

        dRec = DirectSamplingRecord(its);

        if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
            (bsdf->getType() & BSDF::ESmooth)) {
            neeEmitterValue = scene->sampleEmitterDirect(dRec, sample, testVisibility);
            neeEmitterValue *= dRec.pdf;
            
            const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
            if (!neeEmitterValue.isZero()) {
                /* Allocate a record for querying the BSDF */
                BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                /* Evaluate BSDF * cos(theta) */
                const Spectrum bsdfVal = bsdf->eval(bRec);

                /* Prevent light leaks due to the use of shading normals */
                if (!bsdfVal.isZero() && (!m_strictNormals
                        || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                    /* Calculate prob. of having generated that direction
                        using BSDF sampling */
                    lumPdf = dRec.pdf;
                    bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                        ? bsdf->pdf(bRec) : 0;

                    /* Weight using the power heuristic */
                    // mis_weight_nee = miWeight(angle_pdf_nee_as_nee, angle_pdf_nee_as_bsdf, 2);

                    em_path_length = path_length + dRec.dist;

                    G_nee = 1;
                    if(emitter->isOnSurface()){
                        G_nee = std::abs(dot(dRec.d, dRec.n)) / (dRec.dist * dRec.dist);
                    }
                    // path_pdf_nee_as_nee = angle_pdf_nee_as_nee * G_nee;
                    // if(given_path_pdf_nee_as_next_ray >= 0){
                    //     path_pdf_nee_as_next_ray = given_path_pdf_nee_as_next_ray;
                    // } else {
                    //     path_pdf_nee_as_next_ray = angle_pdf_nee_as_bsdf * G_nee;
                    // }
                    path_throughput_nee = path_throughput * bsdfVal;
                }
            }
        }
    }

    void get_next_ray_from_sample(Point2 &sample){
        //path_pdf_next_ray_as_next_ray = 0.0f;
        //path_pdf_next_ray_as_nee = 0.0f;
        bsdfPdf = 0.0;
        lumPdf = 0.0;        
        hitEmitter = false;
        hitEmitterValue = Spectrum(0.0);
        prev_its = its;
        G = 1;
        if(this->path_terminated)
            return;
        
        const BSDF *bsdf = its.getBSDF(ray);

        /* Sample BSDF * cos(theta) */
        
        BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
        Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, sample);
        bsdfVal = bsdfWeight * bsdfPdf;

        const Vector wo = its.toWorld(bRec.wo);
        ray = Ray(its.p, wo, ray.time);

        scene->rayIntersect(ray, next_its);
        bsdf_trace(bRec);
    }

    void get_next_ray_from_its(Intersection& _its){
        //path_pdf_next_ray_as_next_ray = 0.0f;
        //path_pdf_next_ray_as_nee = 0.0f;
        bsdfPdf = 0.0;
        lumPdf = 0.0;        
        hitEmitter = false;
        hitEmitterValue = Spectrum(0.0);
        prev_its = its;
        G = 1;

        this->path_terminated |= !_its.isValid();
        
        if(this->path_terminated)
            return;

        const BSDF *bsdf = its.getBSDF(ray);
        next_its = _its;
        next_its.adjustTime(ray.time);
        //Point target_p = next_its.p;
        Float dist = (next_its.p - its.p).length();

        const Vector wo = normalize(next_its.p - its.p);
        ray = Ray(its.p, wo, ray.time);
        scene->rayIntersect(ray, next_its);

        // next_its.t = (next_its.p - its.p).length();

        // const Vector wo = normalize(next_its.p - its.p);

        BSDFSamplingRecord bRec(its, its.toLocal(wo), ERadiance);
        bsdfVal = bsdf->eval(bRec);
        bsdfPdf = bsdf->pdf(bRec);
        
        // // Check discontinuity
        // Intersection next_its_temp;
        // ray = Ray(its.p, wo, ray.time);
        // ray.maxt = next_its.t*(1-ShadowEpsilon); //std::max(, Epsilon);
        // ray.mint = Epsilon;
        if(dist - 1e-10 > next_its.t){
            this->path_terminated=true;
        }
        // path_terminated = scene->rayIntersect(ray, next_its_temp);
        if(this->path_terminated){
            return;
        }
        // get_sample_from_direction();

        // const BSDF *bsdf = its.getBSDF(ray);
        // sample = bsdf->get_sample_from_direction(bRec);

        // BSDFSamplingRecord bRec2(its, rRec.sampler, ERadiance);
        // Float temp;
        // bsdf->sample(bRec2, temp, sample);
        // const Vector wo_local = its.toLocal(wo);
        bsdf_trace(bRec);
    }

    static void update_path_pdf_positional(PathTracePart &path1, PathTracePart &path2){
        Float jacobian = path2.G / path1.G;
        //path1.set_path_pdf_as(path1, path1.bsdfPdf);
        path1.set_path_pdf_as(path2, path2.bsdfPdf * jacobian);
        path2.set_path_pdf_as(path1, path2.bsdfPdf);
        path2.set_path_pdf_as(path2, path1.bsdfPdf / jacobian);
    }

    static void update_path_pdf_sampler(PathTracePart &path1, PathTracePart &path2){
        //path1.set_path_pdf_as(path1, path1.bsdfPdf);
        path1.set_path_pdf_as(path2, path1.bsdfPdf);
        path2.set_path_pdf_as(path1, path2.bsdfPdf);
        path2.set_path_pdf_as(path2, path2.bsdfPdf);
    }

    static void calculate_next_ray_pdf_positional(PathTracePart &path1, PathTracePart &path2){
        path1.path_pdf_next_ray_as_nee = path1.lumPdf;
        path1.path_pdf_next_ray_as_next_ray = path1.bsdfPdf;

        path2.path_pdf_next_ray_as_nee = path2.lumPdf;
        path2.path_pdf_next_ray_as_next_ray = path1.bsdfPdf * path1.G / path2.G;
    }

    static void calculate_next_ray_pdf_sampler(PathTracePart &path1, PathTracePart &path2){
        path1.path_pdf_next_ray_as_nee = path1.lumPdf;
        path1.path_pdf_next_ray_as_next_ray = path1.bsdfPdf;

        path2.path_pdf_next_ray_as_nee = path2.lumPdf;
        path2.path_pdf_next_ray_as_next_ray = path2.bsdfPdf;
    }

    static void calculate_nee_pdf_positional(PathTracePart &path1, PathTracePart &path2){
        path1.path_pdf_nee_as_nee = path1.lumPdf;
        path1.path_pdf_nee_as_next_ray = path1.bsdfPdf;

        path2.path_pdf_nee_as_nee = path2.lumPdf;
        path2.path_pdf_nee_as_next_ray = path1.bsdfPdf * path1.G_nee / path2.G_nee;
    }

    static void calculate_nee_pdf_sampler(PathTracePart &path1, PathTracePart &path2){
        path1.path_pdf_nee_as_nee = path1.lumPdf;
        path1.path_pdf_nee_as_next_ray = path1.bsdfPdf;

        path2.path_pdf_nee_as_nee = path2.lumPdf;
        path2.path_pdf_nee_as_next_ray = path2.bsdfPdf;
    }


    static void calculate_bsdf_pdf_as_other(PathTracePart &path1, PathTracePart &path2, PathTracePart &path3, Float jacobian){
        //path1.set_path_pdf_as_nee(path1, path1.lumPdf * path1.G);
        //path1.set_path_pdf_as_nee(path2, path2.lumPdf * path2.G * jacobian);
        //path2.set_path_pdf_as_nee(path1, path2.lumPdf * path2.G);
        //path2.set_path_pdf_as_nee(path3, path1.lumPdf * path1.G / jacobian);
        
        path1.set_path_pdf_as(path1, path1.bsdfPdf);
        path1.set_path_pdf_as(path2, path2.bsdfPdf * jacobian);
        path2.set_path_pdf_as(path1, path2.bsdfPdf);
        path2.set_path_pdf_as(path3, path1.bsdfPdf / jacobian);
    }

    void bsdf_trace(const BSDFSamplingRecord& bRec){
        this->wo = bRec.wo;
        const BSDF *bsdf = its.getBSDF(ray);

        /* Sample BSDF * cos(theta) */

        if (bsdfVal.isZero()){
            path_terminated = true;
            return;
        }

        scattered |= bRec.sampledType != BSDF::ENull;

        /* Prevent light leaks due to the use of shading normals */
        const Vector wo = its.toWorld(bRec.wo);
        Float woDotGeoN = dot(its.geoFrame.n, wo);
        if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0){
            path_terminated = true;
            return;
        }

        /* Trace a ray in this direction */
        its = next_its;
        if (its.isValid()) {
            /* Intersected something - check if it was a luminaire */
            if (its.isEmitter()) {
                hitEmitterValue = its.Le(-ray.d);
                dRec.setQuery(ray, its);
                hitEmitter = true;
            }
        } else {
            /* Intersected nothing -- perhaps there is an environment map? */
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env) {
                if (m_hideEmitters && !scattered){
                    path_terminated = true;
                    return;
                }

                hitEmitterValue = env->evalEnvironment(ray);
                if (!env->fillDirectSamplingRecord(dRec, ray)){
                    path_terminated = true;
                    return;
                }
                hitEmitter = true;
            } else {
                path_terminated = true;
                return;
            }
        }
        
        G = (bRec.sampledType & BSDF::EDelta) ? 1.0 : std::abs(dot(its.shFrame.n, ray.d)) / (its.t * its.t);
        path_length += its.t;

        /* Keep track of the throughput and relative
            refractive index along the path */
        path_throughput *= bsdfVal;
        eta *= bRec.eta;

        /* If a luminaire was hit, estimate the local illumination and
            weight using the power heuristic */
        if (hitEmitter &&
            (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
            /* Compute the prob. of generating that direction using the
                implemented direct illumination sampling technique */
            lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                scene->pdfEmitterDirect(dRec) : 0;
            // Li += throughput * value * miWeight(bsdfPdf, lumPdf);
        }
    }
};