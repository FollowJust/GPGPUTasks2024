#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

// essentially, CPU code from nbody_cpu translated to GPU code
__kernel void nbody_calculate_force_global(
    __global float * pxs, __global float * pys,
    __global float *vxs, __global float *vys,
    __global const float *mxs,
    __global float * dvx2d, __global float * dvy2d,
    int N,
    int t)
{
    const unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    const float x0 = pxs[i];
    const float y0 = pys[i];
    const float m0 = mxs[i];

    dvx[i] = 0.0f;
    dvy[i] = 0.0f;
    for (unsigned int j = 0; j < N; ++j)
    {
        // skip ourselves
        if (i == j) 
        {
            continue;
        }

        const float x1 = pxs[j];
        const float y1 = pys[j];
        const float m1 = mxs[j];

        const float dx = x1 - x0;
        const float dy = y1 - y0;

        const float dr2 = max(100.f, dx * dx + dy * dy);

        const float dr2_inv = 1.f / dr2;
        const float dr_inv = sqrt(dr2_inv);

        const float ex = dx * dr_inv;
        const float ey = dy * dr_inv;

        const float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
        const float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

        dvx[i] += m1 * fx;
        dvy[i] += m1 * fy;
    }
}

__kernel void nbody_integrate(
        __global float * pxs, __global float * pys,
        __global float *vxs, __global float *vys,
        __global const float *mxs,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    vxs[i] += dvx[i];
    vys[i] += dvy[i];
    pxs[i] += vxs[i];
    pys[i] += vys[i];
}
