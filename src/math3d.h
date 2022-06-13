#pragma once
#include <common.h>
#include <math.h>

constexpr f64 PI64 = 3.14159265359;
constexpr f32 PI32 = 3.14159265359;

template<typename T>
T ToRadian(T degree) {
    constexpr f64 CONV = PI32 / 180.0f;
    return degree * CONV;
}

template<typename T, u32 n> 
struct vec {
    T arr[n];
    T& operator [](u32 i) {
        return arr[i];
    }

    template<typename To>
    explicit operator To() {
        vec<To, n> ret;
        for(u32 i = 0; i < n; i++) {
            ret[i] = (To)arr[i];
        }
        return ret;
    }

    vec<T, n> operator + (T scalar) {       
        vec<T, n> r;
        for(u32 i = 0; i < n; i++) {
            r[i] = arr[i] + scalar;
        }
        return r;
    }
    vec<T, n> operator - (T scalar) {
        vec<T, n> r;
        for(u32 i = 0; i < n; i++) {
            r[i] = arr[i] - scalar;
        }
        return r;
    }
    vec<T, n> operator * (T scalar) {
        vec<T, n> r;
        for(u32 i = 0; i < n; i++) {
            r[i] = arr[i] * scalar;
        }
        return r;
    }
    vec<T, n> operator / (T scalar) {
        vec<T, n> r;
        auto c = 1 / scalar;
        for(u32 i = 0; i < n; i++) {
            r[i] = arr[i] * c;
        }
        return r;
    }

    vec<T, n> operator + (vec<T, n> other) {

        vec<T, n> r;
        for(u32 i = 0; i < n; i++) {
            r[i] = arr[i] + other[i];
        }
        return r;
    }
    vec<T, n> operator - (vec<T, n> other) {
        vec<T, n> r;
        for(u32 i = 0; i < n; i++) {
            r[i] = arr[i] - other[i];
        }
        return r;
    }
    vec<T, n> operator * (vec<T, n> other) {
        vec<T, n> r;
        for(u32 i = 0; i < n; i++) {
            r[i] = arr[i] * other[i];
        }
        return r;
    }
    vec<T, n> operator / (vec<T, n> other) {
        vec<T, n> r;
        for(u32 i = 0; i < n; i++) {
            r[i] = arr[i] / other[i];
        }
        return r;
    }
    bool operator == (vec<T, n> other) {
        bool r = true;
        for(u32 i = 0; i < n; i++) {
            r &= (arr[i] == other[i]);
        }
        return r;
    }
    bool operator != (vec<T, n> other) {
        bool r = false;
        for(u32 i = 0; i < n; i++) {
            r |= (arr[i] != other[i]);
        }
        return r;
    }
};

template<typename T>
struct vec<T,2>  {
    union {
        T arr[2];
        struct {
            T x;
            T y;
        };
    };
    T& operator [](u32 i) {
        return arr[i];
    }
    template<typename To>
    explicit operator vec<To, 2>() {
        vec<To, 2> ret{(To)x, (To)y};
        return ret;
    }

    vec<T, 2> operator + (T scalar) {
        vec<T, 2> r;
        for(u32 i = 0; i < 2; i++) {
            r[i] = arr[i] + scalar;
        }
        return r;
    }
    vec<T, 2> operator - (T scalar) {
        vec<T, 2> r;
        for(u32 i = 0; i < 2; i++) {
            r[i] = arr[i] - scalar;
        }
        return r;
    }
    vec<T, 2> operator * (T scalar) {
        vec<T, 2> r;
        for(u32 i = 0; i < 2; i++) {
            r[i] = arr[i] * scalar;
        }
        return r;
    }
    vec<T, 2> operator / (T scalar) {
        vec<T, 2> r;
        auto c = 1 / scalar;
        for(u32 i = 0; i < 2; i++) {
            r[i] = arr[i] * c;
        }
        return r;
    }

    vec<T, 2> operator + (vec<T, 2> other) {

        vec<T, 2> r;
        for(u32 i = 0; i < 2; i++) {
            r[i] = arr[i] + other[i];
        }
        return r;
    }
    vec<T, 2> operator - (vec<T, 2> other) {
        vec<T, 2> r;
        for(u32 i = 0; i < 2; i++) {
            r[i] = arr[i] - other[i];
        }
        return r;
    }
    vec<T, 2> operator * (vec<T, 2> other) {
        vec<T, 2> r;
        for(u32 i = 0; i < 2; i++) {
            r[i] = arr[i] * other[i];
        }
        return r;
    }
    vec<T, 2> operator / (vec<T, 2> other) {
        vec<T, 2> r;
        for(u32 i = 0; i < 2; i++) {
            r[i] = arr[i] / other[i];
        }
        return r;
    }
    bool operator == (vec<T,2 > other) {
        bool r = true;
        for(u32 i = 0; i < 2; i++) {
            r &= (arr[i] == other[i]);
        }
        return r;
    }
    bool operator != (vec<T, 2> other) {
        bool r = false;
        for(u32 i = 0; i < 2; i++) {
            r |= (arr[i] != other[i]);
        }
        return r;
    }
};
template<typename T>
struct vec<T,3>  {
    union {
        T arr[3];
        struct {
            T x;
            T y;
            T z;
        };
    };
    T& operator [](u32 i) {
        return arr[i];
    }

    template<typename To>
    explicit operator To() {
        vec<To, 3> ret{(To)x, (To)y, (To)z};
        return ret;
    }

    vec<T, 3> operator + (T scalar) {
        vec<T, 3> r;
        for(u32 i = 0; i < 3; i++) {
            r[i] = arr[i] + scalar;
        }
        return r;
    }
    vec<T, 3> operator - (T scalar) {
        vec<T, 3> r;
        for(u32 i = 0; i < 3; i++) {
            r[i] = arr[i] - scalar;
        }
        return r;
    }
    vec<T, 3> operator * (T scalar) {
        vec<T, 3> r;
        for(u32 i = 0; i < 3; i++) {
            r[i] = arr[i] * scalar;
        }
        return r;
    }
    vec<T, 3> operator / (T scalar) {
        vec<T, 3> r;
        auto c = 1 / scalar;
        for(u32 i = 0; i < 3; i++) {
            r[i] = arr[i] * c;
        }
        return r;
    }

    vec<T, 3> operator + (vec<T, 3> other) {

        vec<T, 3> r;
        for(u32 i = 0; i < 3; i++) {
            r[i] = arr[i] + other[i];
        }
        return r;
    }
    vec<T, 3> operator - (vec<T, 3> other) {
        vec<T, 3> r;
        for(u32 i = 0; i < 3; i++) {
            r[i] = arr[i] - other[i];
        }
        return r;
    }
    vec<T, 3> operator * (vec<T, 3> other) {
        vec<T, 3> r;
        for(u32 i = 0; i < 3; i++) {
            r[i] = arr[i] * other[i];
        }
        return r;
    }
    vec<T, 3> operator / (vec<T, 3> other) {
        vec<T, 3> r;
        for(u32 i = 0; i < 3; i++) {
            r[i] = arr[i] / other[i];
        }
        return r;
    }
    bool operator == (vec<T,3 > other) {
        bool r = true;
        for(u32 i = 0; i < 3; i++) {
            r &= (arr[i] == other[i]);
        }
        return r;
    }
    bool operator != (vec<T, 3> other) {
        bool r = false;
        for(u32 i = 0; i < 3; i++) {
            r |= (arr[i] != other[i]);
        }
        return r;
    }
};
template<typename T>
struct vec<T,4>  {
    union {
        T arr[4];
        struct {
            T x;
            T y;
            T z;
            T w;
        };
    };
    T& operator [](u32 i) {
        return arr[i];
    }
    template<typename To>
    explicit operator To() {
        vec<To, 4> ret{(To)x, (To)y, (To)z, (To)w};
        return ret;
    }
    vec<T, 4> operator + (T scalar) {
        vec<T, 4> r;
        for(u32 i = 0; i < 4; i++) {
            r[i] = arr[i] + scalar;
        }
        return r;
    }
    vec<T, 4> operator - (T scalar) {
        vec<T, 4> r;
        for(u32 i = 0; i < 4; i++) {
            r[i] = arr[i] - scalar;
        }
        return r;
    }
    vec<T, 4> operator * (T scalar) {
        vec<T, 4> r;
        for(u32 i = 0; i < 4; i++) {
            r[i] = arr[i] * scalar;
        }
        return r;
    }
    vec<T, 4> operator / (T scalar) {
        vec<T, 4> r;
        auto c = 1 / scalar;
        for(u32 i = 0; i < 4; i++) {
            r[i] = arr[i] * c;
        }
        return r;
    }

    vec<T, 4> operator + (vec<T, 4> other) {

        vec<T, 4> r;
        for(u32 i = 0; i < 4; i++) {
            r[i] = arr[i] + other[i];
        }
        return r;
    }
    vec<T, 4> operator - (vec<T, 4> other) {
        vec<T, 4> r;
        for(u32 i = 0; i < 4; i++) {
            r[i] = arr[i] - other[i];
        }
        return r;
    }
    vec<T, 4> operator * (vec<T, 4> other) {
        vec<T, 4> r;
        for(u32 i = 0; i < 4; i++) {
            r[i] = arr[i] * other[i];
        }
        return r;
    }
    vec<T, 4> operator / (vec<T, 4> other) {
        vec<T, 4> r;
        for(u32 i = 0; i < 4; i++) {
            r[i] = arr[i] / other[i];
        }
        return r;
    }
    bool operator == (vec<T,4 > other) {
        bool r = true;
        for(u32 i = 0; i < 4; i++) {
            r &= (arr[i] == other[i]);
        }
        return r;
    }
    bool operator != (vec<T, 4> other) {
        bool r = false;
        for(u32 i = 0; i < 4; i++) {
            r |= (arr[i] != other[i]);
        }
        return r;
    }
};

typedef vec<f32,2> v2;
typedef vec<f32,3> v3;
typedef vec<f32,4> v4;

template<typename T, u32 m, u32 n>
struct Mat {
    union {
        vec<T, m> bases[n];
        T elem[m][n];
    };

    vec<T, m>& operator [](u32 i) {
        return bases[i];
    }
    vec<T, m> operator * (vec<T, n> v) {
        vec<T, m> res{};
        for(u32 i = 0; i < n; i++) {
            res = res + (bases[i] * v[i]);
        }
        return res;
    }
    Mat<T, m,n> operator * (Mat<T, m,n> other) {

        Mat<T, m,n> res;
        for(u32 i = 0; i < n; i++) {
            res.bases[i] = (*this) * other[i];
        }
        return res;
    }
};
template<typename T>
using Mat4 = Mat<T, 4,4>;

template<typename T>
T cross(vec<T, 2> l , vec<T, 2> r) {
    return l.x * r.y - l.y * r.x;
}
template<typename T>
vec<T, 3> cross(vec<T, 3> l , vec<T, 3> r) {
    return {
        l[1] * r[2] - l[2] * r[1],
        l[2] * r[0] - l[0] * r[2],
        l[0] * r[1] - l[1] * r[0],
    };
}
template<typename T, u32 n>
T dot(vec<T, n> a, vec<T, n> b) {
    T ret = 0;
    for(u32 i = 0; i < n; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}
template<typename T, u32 n>
T length(vec<T,n> v) {
    T sum = 0;
    for(u32 i = 0; i < n; i++) {
        sum = sum + (v.arr[i] * v.arr[i]);
    }
    return sqrt(sum);
}
template<typename T, u32 n>
vec<T, n> normalize(vec<T, n> v) {
    T l = 1/length(v);
    for(u32 i = 0; i < n; i++) {
        v[i] *= l;
    }
    return v;
}
template<typename T, u32 m, u32 n>
T det(Mat<T, m,n> mat) {
    
}
template<typename T, u32 m, u32 n>
T inverse(Mat<T, m,n> mat) {

}

template<typename T, u32 m, u32 n>
void GaussEliminate(Mat<T, m,n>* mat) {

}
template<typename T, u32 m, u32 n>
void PrintMat(Mat<T, m,n> mat) {
    for(u32 k = 0; k < n; k++) {
        for(u32 i = 0; i < m; i++) {
            global_print("fs", (f64)mat.bases[i].arr[k], ", ");
        }
        global_print("c", '\n');
    }
}

struct Camera {
    vec<f32, 3> position;
    vec<f32, 3> direction;
    vec<f32, 3> vel;
};
void ComputeCameraVelocity(Camera* cam, u8 keys, f32 speed);
void RotateCamera(Camera* cam , f32 vertRotAngle , f32 horizRotAngle);
Mat<f32, 3,3> OrientTo(vec<f32, 3> from, vec<f32, 3> to, vec<f32, 3> worldUp);
Mat4<f32> LookAt(vec<f32, 3> from, vec<f32, 3> to, vec<f32, 3> worldUp = {0,1,0});
Mat4<f32> ComputePerspectiveMat4(f32 fov , f32 aspect , f32 near , f32 far);
Mat<f32, 3,3> ComputeRotarionXMat4(f32 x);
Mat<f32, 3,3> ComputeRotarionYMat4(f32 x);
Mat<f32, 3,3> ComputeRotarionZMat4(f32 x);
vec<f32, 3> ComputeRotateAroundAxis(vec<f32, 3> axis, vec<f32, 3> v, f32 angle);

