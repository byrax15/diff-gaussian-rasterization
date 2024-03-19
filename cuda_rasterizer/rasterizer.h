/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <CullOperator.h>
#include <functional>
#include <stdexcept>
#include <vector>

namespace CudaRasterizer {
struct render_error : std::runtime_error {
    using runtime_error::runtime_error;
};

class Rasterizer {
public:
    static void markVisible(
        int P,
        float* means3D,
        float* viewmatrix,
        float* projmatrix,
        bool* present);

    static int forward(
        std::function<char*(size_t)> geometryBuffer,
        std::function<char*(size_t)> binningBuffer,
        std::function<char*(size_t)> imageBuffer,
        const int P, int D, int M,
        const float* background,
        const int width, int height,
        const float* means3D,
        const float* shs,
        const float* colors_precomp,
        const float* opacities,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const float* cam_pos,
        const float tan_fovx, float tan_fovy,
        const bool prefiltered,
        float* out_color,
        int* radii = nullptr,
        int* rects = nullptr,
        int boxcount = 0,
        const float* boxmin = nullptr,
        const float* boxmax = nullptr,
        FORWARD::Cull::Operator cullop = FORWARD::Cull::Operator::AND);

    static void backward(
        const int P, int D, int M, int R,
        const float* background,
        const int width, int height,
        const float* means3D,
        const float* shs,
        const float* colors_precomp,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const float* campos,
        const float tan_fovx, float tan_fovy,
        const int* radii,
        char* geom_buffer,
        char* binning_buffer,
        char* image_buffer,
        const float* dL_dpix,
        float* dL_dmean2D,
        float* dL_dconic,
        float* dL_dopacity,
        float* dL_dcolor,
        float* dL_dmean3D,
        float* dL_dcov3D,
        float* dL_dsh,
        float* dL_dscale,
        float* dL_drot);

    struct GaussianProperties {
        float* pos_cuda {};
        float* rot_cuda {};
        float* scale_cuda {};
        float* opacity_cuda {};
        float* shs_cuda {};
    };
    struct GaussianScene {
        size_t start_index {}, count {};
        float3 position {};
        float opacity = 1.f;

        GaussianScene() = default;
        GaussianScene(size_t start_index, size_t count)
            : start_index(start_index)
            , count(count)
        {
        }
    };
    static void sceneToWorldAsync(GaussianProperties scene_space, GaussianProperties world_space, GaussianScene* scenes, size_t scene_count);
};
};

#endif