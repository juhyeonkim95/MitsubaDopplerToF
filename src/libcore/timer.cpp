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

#include <mitsuba/core/timer.h>

#if defined(_MSC_VER)
// Silent the "unary minus applied to unsigned" warning generated by rational
# pragma warning (disable : 4146)
#endif

#if defined(__WINDOWS__)
# include <windows.h>
#elif defined(__OSX__)
// Info from the Mac OS X Reference Library (accessed March 2012)
// http://developer.apple.com/library/mac/#qa/qa1398/_index.html
// https://developer.apple.com/library/mac/#documentation/Darwin/Conceptual/KernelProgramming/services/services.html
# include <CoreServices/CoreServices.h>
# include <mach/mach.h>
# include <mach/mach_time.h>
# include <unistd.h>
#else
// Assume POSIX. Check for good clock sources
# include <unistd.h>
# include <ctime>
# if _POSIX_C_SOURCE < 199309L
# error "The required POSIX clock functions are not available."
# endif
# if defined(_POSIX_MONOTONIC_CLOCK)
# define TIMER_CLOCK CLOCK_MONOTONIC
# elif defined(CLOCK_HIGHRES)
# define TIMER_CLOCK CLOCK_HIGHRES
# elif defined(CLOCK_REALTIME)
# define TIMER_CLOCK CLOCK_REALTIME
# else
# error "No suitable clock found.  Check docs for clock_gettime."
# endif
#endif

MTS_NAMESPACE_BEGIN

namespace {
    #if !defined(__WINDOWS__) && !defined(__OSX__)
        inline double timespecToNano(const timespec &x) {
            return x.tv_sec * 1e9 + x.tv_nsec;
        }
    #endif

    /// Return the resolution of the native time measure expressed in nanoseconds/tick
    static double timerResolution() {
        #if defined(__WINDOWS__)
            LARGE_INTEGER freq;
            if (!QueryPerformanceFrequency(&freq))
                SLog(EError, "Could not query the high performance counter: %s",
                    lastErrorText().c_str());
            return 1.0 / static_cast<double>(freq.QuadPart) * 1e9;
        #elif defined(__OSX__)
            mach_timebase_info_data_t info;
            mach_timebase_info(&info);
            return (double) info.numer / (double) info.denom;
        #else
            return 1.0;
        #endif
    }

    /// Invoke timerResolution() once and store the result
    static double __resolution = timerResolution();

    /// Current time snapshot in nanoseconds
    inline double timeInNanoseconds() {
        #if defined(__WINDOWS__)
            LARGE_INTEGER perfcount;
            QueryPerformanceCounter(&perfcount);
            return perfcount.QuadPart * __resolution;
        #elif defined(__OSX__)
            return mach_absolute_time() * __resolution;
        #else
            timespec tspec;
            clock_gettime(TIMER_CLOCK, &tspec);
            return timespecToNano(tspec);
        #endif
    }
}

Timer::Timer(bool startNow) : m_elapsed(0), m_active(false) {
    if (startNow)
        start();
}

Timer::~Timer() {
}

void Timer::start() {
    if (!m_active) {
        m_startTime = timeInNanoseconds();
        m_active   = true;
    }
#if defined(MTS_DEBUG)
    else {
        Log(EWarn, "The timer is already active, ignoring start()");
    }
#endif
}

Float Timer::stop() {
    if (m_active) {
        m_elapsed += timeInNanoseconds() - m_startTime;
        m_active = false;
    }
#if defined(MTS_DEBUG)
    else {
        Log(EWarn, "The timer is not active, ignoring stop()");
    }
#endif
    return (Float) (m_elapsed * 1e-9);
}

void Timer::reset(bool restart) {
    m_elapsed = 0;
    m_active = false;
    if (restart)
        start();
}

double Timer::timeSinceStart() const {
    if (m_active)
        return timeInNanoseconds() - m_startTime;
    else
        return 0.0;
}

uint64_t Timer::getNanoseconds() const {
    return (uint64_t) (timeSinceStart() + m_elapsed);
}

unsigned int Timer::getMicroseconds() const {
    return (unsigned int) ((timeSinceStart() + m_elapsed) * 1e-3);
}

unsigned int Timer::getMilliseconds() const {
    return (unsigned int) ((timeSinceStart() + m_elapsed) * 1e-6);
}

Float Timer::getSeconds() const {
    return (Float) ((timeSinceStart() + m_elapsed) * 1e-9);
}

uint64_t Timer::getNanosecondsSinceStart() const {
    return (uint64_t) timeSinceStart();
}

unsigned int Timer::getMicrosecondsSinceStart() const {
    return (unsigned int) (timeSinceStart() * 1e-3);
}

unsigned int Timer::getMillisecondsSinceStart() const {
    return (unsigned int) (timeSinceStart() * 1e-6);
}

Float Timer::getSecondsSinceStart() const {
    return (Float) (timeSinceStart() * 1e-9);
}

Float Timer::lap() {
    double time = timeInNanoseconds();
    double delta = m_active ? (time - m_startTime) : 0;
    m_elapsed += delta;
    m_startTime = time;
    m_active = true;
    return (Float) (delta * 1e-9);
}

std::string Timer::toString() const {
    std::ostringstream oss;
    oss << "Timer[ms=" << getMilliseconds() << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS(Timer, false, Object)
MTS_NAMESPACE_END
