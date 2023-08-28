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

static StatsCounter avgPathLength("Radial Velocity Approximator", "Average path length", EAverage);

class RadialVelocityApproximator : public MonteCarloIntegrator {
public:
    RadialVelocityApproximator(const Properties &props)
        : MonteCarloIntegrator(props) {
            m_offset = props.getFloat("image_offset", 100); // output can be negative, so add offset to make it positive
        }

    /// Unserialize from a binary data stream
    RadialVelocityApproximator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        RadianceQueryRecord rRec2 = rRec;

        RayDifferential ray1;
        rRec.scene->getSensor()->sampleRayDifferential(
            ray1, rRec.samplePos, rRec.apertureSample, 0.0);
        ray1.scaleDifferential(rRec.diffScaleFactor);
        
        RayDifferential ray2;
        rRec.scene->getSensor()->sampleRayDifferential(
            ray2, rRec.samplePos, rRec.apertureSample, 1.0);
        ray2.scaleDifferential(rRec.diffScaleFactor);

        rRec.rayIntersect(ray1);
        rRec2.rayIntersect(ray2);
        
        if(rRec.its.isValid() && rRec2.its.isValid()){
            Float d1 = rRec.its.t;
            Float d2 = rRec2.its.t;
            Float v = (d1 - d2) / (ray1.time - ray2.time);
            return Spectrum(m_offset + v);
        }
        return Spectrum(m_offset);
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
        oss << "RadialVelocityApproximator[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(RadialVelocityApproximator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(RadialVelocityApproximator, "MI path tracer");
MTS_NAMESPACE_END
