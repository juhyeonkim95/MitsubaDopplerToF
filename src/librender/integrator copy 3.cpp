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

#include <mitsuba/core/statistics.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>

MTS_NAMESPACE_BEGIN

Integrator::Integrator(const Properties &props)
 : NetworkedObject(props) { }

Integrator::Integrator(Stream *stream, InstanceManager *manager)
 : NetworkedObject(stream, manager) { }

bool Integrator::preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) { return true; }
void Integrator::postprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) { }
void Integrator::serialize(Stream *stream, InstanceManager *manager) const {
    NetworkedObject::serialize(stream, manager);
}
void Integrator::configureSampler(const Scene *scene, Sampler *sampler) {
    /* Prepare the sampler for bucket-based rendering */
    sampler->setFilmResolution(scene->getFilm()->getCropSize(),
        getClass()->derivesFrom(MTS_CLASS(SamplingIntegrator)));
}
const Integrator *Integrator::getSubIntegrator(int idx) const { return NULL; }

SamplingIntegrator::SamplingIntegrator(const Properties &props)
 : Integrator(props) {
    m_position_direction_error_preprocess_portion = props.getFloat("position_direction_error_preprocess_portion", 0.0);
    m_position_direction_error_texture = new ConstantSpectrumTexture(props.getSpectrum("position_direction_error_texture", Spectrum(.5f)));
}

void SamplingIntegrator::addChild(const std::string &name, ConfigurableObject *child) {
    if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
            && (name == "position_direction_error_texture")) {
        m_position_direction_error_texture = static_cast<Texture *>(child);
    }
}

SamplingIntegrator::SamplingIntegrator(Stream *stream, InstanceManager *manager)
 : Integrator(stream, manager) { }

void SamplingIntegrator::serialize(Stream *stream, InstanceManager *manager) const {
    Integrator::serialize(stream, manager);
}

Spectrum SamplingIntegrator::E(const Scene *scene, const Intersection &its,
        const Medium *medium, Sampler *sampler, int nSamples, bool handleIndirect) const {
    Spectrum E(0.0f);
    RadianceQueryRecord query(scene, sampler);
    DirectSamplingRecord dRec(its);
    Frame frame(its.shFrame.n);

    sampler->generate(Point2i(0));
    for (int i=0; i<nSamples; i++) {
        /* Sample the direct illumination component */
        int maxIntermediateInteractions = -1;
        Spectrum directRadiance = scene->sampleAttenuatedEmitterDirect(
            dRec, its, medium, maxIntermediateInteractions, query.nextSample2D());

        if (!directRadiance.isZero()) {
            Float dp = dot(dRec.d, its.shFrame.n);
            if (dp > 0)
                E += directRadiance * dp;
        }

        /* Sample the indirect illumination component */
        if (handleIndirect) {
            query.newQuery(RadianceQueryRecord::ERadianceNoEmission, medium);
            Vector d = frame.toWorld(warp::squareToCosineHemisphere(query.nextSample2D()));
            ++query.depth;
            query.medium = medium;
            E += Li(RayDifferential(its.p, d, its.time), query) * M_PI;
        }

        sampler->advance();
    }

    return E / (Float) nSamples;
}

void SamplingIntegrator::cancel() {
    if (m_process)
        Scheduler::getInstance()->cancel(m_process);
}

bool SamplingIntegrator::render(Scene *scene,
        RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) {
    ref<Scheduler> sched = Scheduler::getInstance();
    ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
    ref<Film> film = sensor->getFilm();

    size_t nCores = sched->getCoreCount();
    const Sampler *sampler = static_cast<const Sampler *>(sched->getResource(samplerResID, 0));
    size_t sampleCount = sampler->getSampleCount();

    Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
        " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
        sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
        nCores == 1 ? "core" : "cores");

    /* This is a sampling-based integrator - parallelize */
    ref<ParallelProcess> proc = new BlockedRenderProcess(job,
        queue, scene->getBlockSize());
    int integratorResID = sched->registerResource(this);
    proc->bindResource("integrator", integratorResID);
    proc->bindResource("scene", sceneResID);
    proc->bindResource("sensor", sensorResID);
    proc->bindResource("sampler", samplerResID);
    scene->bindUsedResources(proc);
    bindUsedResources(proc);
    sched->schedule(proc);

    m_process = proc;
    sched->wait(proc);
    m_process = NULL;
    sched->unregisterResource(integratorResID);

    return proc->getReturnStatus() == ParallelProcess::ESuccess;
}

void SamplingIntegrator::bindUsedResources(ParallelProcess *) const {
    /* Do nothing by default */
}

void SamplingIntegrator::wakeup(ConfigurableObject *parent,
    std::map<std::string, SerializableObject *> &) {
    /* Do nothing by default */
}
inline Float miWeight(Float pdfA, Float pdfB, float power=1) {
    if(pdfA + pdfB == 0.0f){
        return 0.0f;
    }
    float ap = (pdfA == 0.0)? 0.0 : std::pow(pdfA, power);
    float bp = (pdfB == 0.0)? 0.0 : std::pow(pdfB, power);
    return ap / (ap + bp);
}
void SamplingIntegrator::renderBlock(const Scene *scene,
        const Sensor *sensor, Sampler *sampler, ImageBlock *block,
        const bool &stop, const std::vector< TPoint2<uint8_t> > &points) const {

    Float diffScaleFactor = 1.0f /
        std::sqrt((Float) sampler->getSampleCount());

    bool needsApertureSample = sensor->needsApertureSample();
    bool needsTimeSample = sensor->needsTimeSample();

    RadianceQueryRecord rRec(scene, sampler);
    Point2 apertureSample(0.5f);
    Float timeSample = 0.5f;
    RayDifferential sensorRay;

    block->clear();

    uint32_t queryType = RadianceQueryRecord::ESensorRay;

    bool useAntitheticSampling = sensor->useAntitheticSampling();
    bool usePixelCorrelation = sensor->usePixelCorrelation();
    bool useSamplerCorrelation = sensor->useSamplerCorrelation();
    bool isAntitheticSamplingByShift = sensor->isAntitheticSamplingByShift();
    float antitheticShift = sensor->getAntitheticShift();
    
    Float previousTimeSample = 0.0f;
    Point2 previousSamplePos;

    if (!sensor->getFilm()->hasAlpha()) /* Don't compute an alpha channel if we don't have to */
        queryType &= ~RadianceQueryRecord::EOpacity;

    for (size_t i = 0; i<points.size(); ++i) {
        Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
        if (stop)
            break;

        rRec.sampler->generate(offset);

        Float positional_correlation_probability;
        Float q_pos = 1.0f;
        Float q_dir = 1.0f;
        Float n_pos = 0.0f;
        Float n_dir = 0.0f;
        Float alpha = 10.0f;

        Float epsilon = 1.0f;
        Float epsilon_decay = 0.8f;

        for (size_t j = 0; j<rRec.sampler->getSampleCount(); j++) {
            rRec.newQuery(queryType, sensor->getMedium());

            if(rRec.nextSample1D() < epsilon){
                positional_correlation_probability = 0.5f;
            } else {
                Float q_pos_avg = q_pos / n_pos;
                Float q_dir_avg = q_dir / n_dir;
                q_pos_avg = std::pow(q_pos_avg, 50.0);
                q_dir_avg = std::pow(q_dir_avg, 50.0);
                positional_correlation_probability = q_dir_avg / (q_pos_avg + q_dir_avg);
                // positional_correlation_probability = q_dir_avg / (q_pos_avg + q_dir_avg);
            }
            if(rRec.nextSample1D() < positional_correlation_probability){
                rRec.use_positional_correlation_probability = 1.0;
            } else {
                rRec.use_positional_correlation_probability = 0.0;
            }
            if(j > 32){
                epsilon *= epsilon_decay;
            }

            // check using antithetic sampling
            if(useAntitheticSampling){
                if(j%2 == 1){
                    if(isAntitheticSamplingByShift){
                        timeSample = antitheticShift + previousTimeSample;
                        if(timeSample > 1.0f){
                            timeSample -= 1.0f;
                        }
                    } else {
                        timeSample = 1.0f - previousTimeSample;
                    }
                    if(useSamplerCorrelation){
                        rRec.sampler->loadSavedState();
                    }
                } else {
                    if(sensor->hasTimeSampler()){
                    timeSample = sensor->sampleTimeStamp(j, rRec.nextSample1D());
                    } else {
                        timeSample = rRec.nextSample1D();
                    }    
                    previousTimeSample = timeSample;
                    if(useSamplerCorrelation){
                        rRec.sampler->saveState();
                    }
                }
            } else {
                if(sensor->hasTimeSampler()){
                    timeSample = sensor->sampleTimeStamp(j, rRec.nextSample1D());
                } else {
                    timeSample = rRec.nextSample1D();
                }
            }

            Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));
            if(useAntitheticSampling && j%2 == 1 && usePixelCorrelation){
                samplePos = previousSamplePos;
            }

            previousSamplePos = samplePos;


            if (needsApertureSample)
                apertureSample = rRec.nextSample2D();
            
            rRec.samplePos = samplePos;
            rRec.apertureSample = apertureSample;
            rRec.timeSample = timeSample;
            rRec.diffScaleFactor = diffScaleFactor;
            rRec.offset = Point2(offset);
            rRec.sampleIndex = j;
            // rRec.use_positional_correlation_probability = use_positional_correlation_probability;
            
            Spectrum spec = sensor->sampleRayDifferential(
                sensorRay, samplePos, apertureSample, timeSample);

            sensorRay.scaleDifferential(diffScaleFactor);

            spec *= Li(sensorRay, rRec);

            Float timePDF = 1.0;
            if(needsTimeSample){
                timePDF = sensor->pdfTime(sensorRay, ELength);
            }

            Float variance = spec[0] * spec[0] + spec[1] * spec[1] + spec[2] * spec[2];

            spec /= timePDF;
            
            if(rRec.use_positional_correlation_probability == 1.0){
                q_pos += variance;
                //q_pos = variance * alpha + q_pos * (1 - alpha);
                n_pos += 1;
            } else {
                q_dir += variance;
                //q_dir = variance * alpha + q_dir * (1 - alpha);
                n_dir += 1;
            }


            if(this->m_needOffset){
                Spectrum Li_offset(1.0f);
                // spec = Spectrum(use_positional_correlation_probability);
                block->put(samplePos, Li_offset + spec, rRec.alpha);
            } else {
                block->put(samplePos, spec, rRec.alpha);
            }
            
            rRec.sampler->advance();
            if(useAntitheticSampling && usePixelCorrelation){
                if(j%2 == 1){
                    rRec.sampler->advance();
                }
            }
        }
        
        // printf("1. %f\n", q_pos);
        // printf("2. %f\n", q_dir);
        // printf("3. %f\n", n_pos);
        // printf("4. %f\n", n_dir);
        

    }
}

MonteCarloIntegrator::MonteCarloIntegrator(const Properties &props) : SamplingIntegrator(props) {
    /* Depth to begin using russian roulette */
    m_rrDepth = props.getInteger("rrDepth", 5);

    /* Longest visualized path depth (\c -1 = infinite).
       A value of \c 1 will visualize only directly visible light sources.
       \c 2 will lead to single-bounce (direct-only) illumination, and so on. */
    m_maxDepth = props.getInteger("maxDepth", -1);

    /**
     * This parameter specifies the action to be taken when the geometric
     * and shading normals of a surface don't agree on whether a ray is on
     * the front or back-side of a surface.
     *
     * When \c strictNormals is set to \c false, the shading normal has
     * precedence, and rendering proceeds normally at the risk of
     * introducing small light leaks (this is the default).
     *
     * When \c strictNormals is set to \c true, the random walk is
     * terminated when encountering such a situation. This may
     * lead to silhouette darkening on badly tesselated meshes.
     */
    m_strictNormals = props.getBoolean("strictNormals", false);

    /**
     * When this flag is set to true, contributions from directly
     * visible emitters will not be included in the rendered image
     */
    m_hideEmitters = props.getBoolean("hideEmitters", false);

    if (m_rrDepth <= 0)
        Log(EError, "'rrDepth' must be set to a value greater than zero!");

    if (m_maxDepth <= 0 && m_maxDepth != -1)
        Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");
}

MonteCarloIntegrator::MonteCarloIntegrator(Stream *stream, InstanceManager *manager)
    : SamplingIntegrator(stream, manager) {
    m_rrDepth = stream->readInt();
    m_maxDepth = stream->readInt();
    m_strictNormals = stream->readBool();
    m_hideEmitters = stream->readBool();
}

void MonteCarloIntegrator::serialize(Stream *stream, InstanceManager *manager) const {
    SamplingIntegrator::serialize(stream, manager);
    stream->writeInt(m_rrDepth);
    stream->writeInt(m_maxDepth);
    stream->writeBool(m_strictNormals);
    stream->writeBool(m_hideEmitters);
}

std::string RadianceQueryRecord::toString() const {
    std::ostringstream oss;
    oss << "RadianceQueryRecord[" << endl
        << "  type = { ";
    if (type & EEmittedRadiance) oss << "emitted ";
    if (type & ESubsurfaceRadiance) oss << "subsurface ";
    if (type & EDirectSurfaceRadiance) oss << "direct ";
    if (type & EIndirectSurfaceRadiance) oss << "indirect ";
    if (type & ECausticRadiance) oss << "caustic ";
    if (type & EDirectMediumRadiance) oss << "inscatteredDirect ";
    if (type & EIndirectMediumRadiance) oss << "inscatteredIndirect ";
    if (type & EDistance) oss << "distance ";
    if (type & EOpacity) oss << "opacity ";
    if (type & EIntersection) oss << "intersection ";
    oss << "}," << endl
        << "  depth = " << depth << "," << endl
        << "  its = " << indent(its.toString()) << endl
        << "  alpha = " << alpha << "," << endl
        << "  extra = " << extra << "," << endl
        << "]" << endl;
    return oss.str();
}


MTS_IMPLEMENT_CLASS(Integrator, true, NetworkedObject)
MTS_IMPLEMENT_CLASS(SamplingIntegrator, true, Integrator)
MTS_IMPLEMENT_CLASS(MonteCarloIntegrator, true, SamplingIntegrator)
MTS_NAMESPACE_END
