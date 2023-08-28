#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>

using namespace mitsuba;

enum EWaveformType {
    WAVE_TYPE_SINUSOIDAL = 0,
    WAVE_TYPE_RECTANGULAR = 1,
    WAVE_TYPE_TRIANGULAR = 2,
    WAVE_TYPE_TRAPEZOIDAL = 3
};

// s(t) : sensor modulation
// g(t) : illumination modulation
// L(t) : low-pass(s(t) * g(t))

// Eval s(t) or g(t)
Float evalModulationFunctionValue(EWaveformType wave_function_type, Float _t) {
    Float t = std::fmod(_t, 2 * M_PI);
    switch(wave_function_type){
        case WAVE_TYPE_SINUSOIDAL: return std::cos(t);
        case WAVE_TYPE_RECTANGULAR: return std::abs(t-M_PI) > 0.5 * M_PI ? 1 : -1;
        case WAVE_TYPE_TRIANGULAR: return t < M_PI ? 1 - 2 * t / M_PI : -3 + 2 * t / M_PI;
    }
    return std::cos(t);
}

// Eval L(t)
Float evalModulationFunctionValueLowPass(EWaveformType wave_function_type, Float _t) {
    Float t = std::fmod(_t, 2 * M_PI);
    switch(wave_function_type){
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

// integrate L(a * t + b) * (1 + c * t) over t
// Only implemented for sinusoidal now!
Float evalModulationFunctionValueLowPassIntegrated(EWaveformType wave_function_type, Float a, Float b, Float c, Float t){
    // avoid too small value
    if(std::abs(a) < 1e-3){
        Float B_0 = std::cos(b) * t;
        Float B_1 = 0.5 * c * t * t * std::cos(b);
        return B_0 + B_1;
    } else {
        Float A_0 = std::sin(a * t + b) / a;
        Float A_1 = ( c * t * std::sin(a * t + b) / a + c * std::cos(a * t + b) / (a * a));
        return A_0 + A_1;
    }
}
