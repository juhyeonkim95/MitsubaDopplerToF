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

#pragma once
#if !defined(__MITSUBA_RENDER_TOFINTEGRATOR_H_)
#define __MITSUBA_RENDER_TOFINTEGRATOR_H_

#include <mitsuba/core/netobject.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/shape.h>

MTS_NAMESPACE_BEGIN


/*
 * \brief Base class of all recursive Monte Carlo integrators, which compute
 * unbiased solutions to the rendering equation (and optionally
 * the radiative transfer equation).
 * \ingroup librender
 */
class MTS_EXPORT_RENDER ToFPathCorrelatedIntegrator : public MonteCarloIntegrator {
public:
    /// Serialize this integrator to a binary data stream
    // void serialize(Stream *stream, InstanceManager *manager) const;
    Spectrum trace_two_rays(
        const RayDifferential &r1,
        const RayDifferential &r2, 
        RadianceQueryRecord &rRec
    ) const;
    virtual Spectrum accumulate_next_ray_Li(PathTracePart &path1, PathTracePart &path2) const;
    virtual Spectrum accumulate_nee_Li(PathTracePart &path1, PathTracePart &path2) const;
    MTS_DECLARE_CLASS()
protected:
    /// Create a integrator
    ToFPathCorrelatedIntegrator(const Properties &props);
    /// Unserialize an integrator
    ToFPathCorrelatedIntegrator(Stream *stream, InstanceManager *manager);
    /// Virtual destructor
    virtual ~ToFPathCorrelatedIntegrator() { }
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_TOFINTEGRATOR_H_ */