#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>

using namespace mitsuba;

bool check_consistency(Intersection& its1, Intersection& its2){
    bool consistent = its1.isValid() && its2.isValid() && (its1.p - its2.p).length() < 0.1 && dot(its1.shFrame.n, its2.shFrame.n) > 0.7;
    return consistent;
}

void printPoint(Point p1, Point p2){
    printf("P1: %f, %f, %f / P2: %f, %f, %f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
}
void printVector(Vector p1, Vector p2){
    printf("V1: %f, %f, %f / V2: %f, %f, %f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
}

inline Float miWeight(Float pdfA, Float pdfB, float power=1) {
    if(pdfA + pdfB == 0.0f){
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


float getClosestDiscontinuityDistance(Intersection &its) {
    if(!its.isValid()){
        return 0.0;
    }
    if(its.instance){
        // const Instance *instance = reinterpret_cast<const Instance*>(next_its_p.instance);
        // Ray ray_local;
        // Transform trafo = instance->getAnimatedTransform()->eval(ray.time);
        // trafo.inverse(ray, ray_local);
        std::string shape_name = typeid(*its.shape).name();
        
        if (shape_name.find("Rectangle") != std::string::npos){
            //next_its_p.shape->rayIntersectForced(ray_temp, ray_temp.mint, ray_temp.maxt, t_temp, temp);
            //next_its_d.p = trafo(ray_temp(t_temp));
            return 1.0;
        } else {
            Vector e0 = (its.p1w - its.p2w);
            Vector e1 = (its.p2w - its.p0w);
            Vector e2 = (its.p0w - its.p1w);
            float area = 0.5 * cross(e0, e1).length();
            float h0 = area / e0.length() * its.barycentric[0];
            float h1 = area / e1.length() * its.barycentric[1];
            float h2 = area / e2.length() * its.barycentric[2];

            return area;//std::min(h0, std::min(h1, h2));
        }
    } else {
        return 1.0;
    }
}