#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>
#include <sstream>

using namespace mitsuba;

void printPoint(Point p1, Point p2){
    printf("P1: %f, %f, %f / P2: %f, %f, %f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
}
void printVector(Vector p1, Vector p2){
    printf("V1: %f, %f, %f / V2: %f, %f, %f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
}

inline Float miWeight(Float pdfA, Float pdfB, float power=1) {
    if(pdfA == 0.0f && pdfB == 0.0){
        return 0.0f;
    }
    float ap = (pdfA == 0.0)? 0.0 : std::pow(pdfA, power);
    float bp = (pdfB == 0.0)? 0.0 : std::pow(pdfB, power);
    return ap / (ap + bp);
}

inline Float miWeight4(Float pdfA, Float pdfB, Float pdfC, Float pdfD, float power=1) {
    if(pdfA + pdfB +pdfC + pdfD == 0.0f){
        return 0.0f;
    }
    float ap = std::pow(pdfA, power);
    float bp = std::pow(pdfB, power);
    float cp = std::pow(pdfC, power);
    float dp = std::pow(pdfD, power);
    return ap / (ap + bp + cp + dp);
}


std::vector<float> parse_float_array_from_string (std::string &numbers) {

    // If possible, always prefer std::vector to naked array
    std::vector<float> v;

    // Build an istream that holds the input string
    std::istringstream iss(numbers);

    // Iterate over the istream, using >> to grab floats
    // and push_back to store them in the vector
    std::copy(std::istream_iterator<float>(iss),
        std::istream_iterator<float>(),
        std::back_inserter(v));

    // Put the result on standard out
    std::copy(v.begin(), v.end(),
        std::ostream_iterator<float>(std::cout, ", "));
    std::cout << "\n";

    return v;
}