#include <math3d.h>

void ComputeCameraVelocity(Camera* cam, u8 keys, f32 speed) {

    u8 w = ((keys >> 0) & 1);
    u8 a = ((keys >> 1) & 1);
    u8 s = ((keys >> 2) & 1);
    u8 d = ((keys >> 3) & 1);
    u8 space = ((keys >> 4) & 1);
    u8 shift = ((keys >> 5) & 1);
   
    vec<f32, 2> horizontalForwardDir{cam->direction.x ,cam->direction.z};
    horizontalForwardDir = normalize(horizontalForwardDir);
    vec<f32, 2> horizontalOrtoDir{horizontalForwardDir.y , -horizontalForwardDir.x};

    i8 forward = w-s;
    i8 ortogonal = a-d;
    i8 vertical = space-shift;

    cam->vel.y += vertical * speed;
    cam->vel.x += (horizontalForwardDir.x * forward * speed) + (horizontalOrtoDir.x * ortogonal * speed);
    cam->vel.z += (horizontalForwardDir.y * forward * speed) + (horizontalOrtoDir.y * ortogonal * speed);
}

void RotateCamera(Camera* cam , f32 vertRotAngle , f32 horizRotAngle) {

    f32 cosHoriz = cos(horizRotAngle);
    f32 sinHoriz = sin(horizRotAngle);

    f32 cosVert = cos(vertRotAngle);
    f32 sinVert = sin(vertRotAngle);

    cam->direction.x = cam->direction.x * cosHoriz - cam->direction.z * sinHoriz;
    cam->direction.z = cam->direction.x * sinHoriz + cam->direction.z * cosHoriz;

    cam->direction = normalize(cam->direction);
    vec<f32, 3> right = normalize(cross(cam->direction, {0,1,0}));
    vec<f32, 3> w = normalize(cross(right, cam->direction));

    cam->direction = cam->direction * cosVert + w * sinVert;
    cam->direction = normalize(cam->direction);
}

Mat4<f32> LookAt(vec<f32, 3> from, vec<f32, 3> to, vec<f32, 3> worldUp) {

    vec<f32, 3> forward{ normalize(to - from) };
    vec<f32, 3> right{ normalize(cross(forward, worldUp)) };
    vec<f32, 3> up{ normalize(cross(right, forward)) };

    return Mat4<f32> {
        right.x , up.x , -forward.x , 0,
        right.y , up.y , -forward.y , 0,
        right.z , up.z , -forward.z , 0,
        -dot(right, from), -dot(up, from), dot(forward, from) , 1
    };
}

Mat<f32, 3,3> OrientTo(vec<f32, 3> from, vec<f32, 3> to, vec<f32, 3> worldUp) {

    vec<f32, 3> m_Z = normalize(to - from);
    vec<f32, 3> m_X = normalize(cross(worldUp, m_Z));
    vec<f32, 3> m_Y = normalize(cross(m_Z, m_X));

    return Mat<f32, 3,3> {
        m_X.x, m_X.y, m_X.z,
        m_Y.x, m_Y.y, m_Y.z,
        m_Z.x, m_Z.y, m_Z.z,
    };
}

Mat4<f32> ComputePerspectiveMat4(f32 fov , f32 aspect , f32 near , f32 far) {

    f32 tanFov = tan( fov * 0.5 );
    f32 x = 1 / ( aspect * tanFov );
    f32 y = 1 / ( tanFov );

    f32 z = -(far + near) / (far - near);
    f32 w = (-2 * far * near) / (far - near);

    return Mat4<f32> {
        x,0,0,0,
        0,-y,0,0,
        0,0,z,-1,
        0,0,w,0
    };
}
Mat<f32, 3,3> ComputeRotarionXMat4(f32 x) {

    Mat<f32, 3, 3> rot {
        1,0,0,
        0,cos(x),-sin(x),
        0,sin(x),cos(x),
    };
    return rot;
}
Mat<f32, 3,3> ComputeRotarionYMat4(f32 x) {

    Mat<f32, 3,3> rot {
        cos(x),0,sin(x),
        0,1,0,
        -sin(x),0,cos(x),
    };
    return rot;
}
Mat<f32, 3,3> ComputeRotarionZMat4(f32 x) {

    Mat<f32, 3,3> rot {
        cos(x),-sin(x),0,
        sin(x),cos(x),0,
        0,0,1,
    };
    return rot;
}
vec<f32, 3> ComputeRotateAroundAxis(vec<f32, 3> axis, vec<f32, 3> v, f32 angle) {

    auto y = sin(angle);
    auto x = cos(angle);

    auto len = dot(axis, v);
    auto basisZ = axis * len;
    auto basisX = v - basisZ;

    auto basisXLen = length(basisX);
    auto inverseLen = (1.0f / basisXLen);

    auto basisY = cross(axis, basisX * inverseLen) * basisXLen;

    return basisZ + basisX * x + basisY * y;
}

