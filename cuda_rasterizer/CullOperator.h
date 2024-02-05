#pragma once
#include <array>
#include <numeric>
#include <type_traits>
#include "cuda_runtime.h"

namespace FORWARD {
	namespace Cull {
		enum class Operator { AND, OR, XOR };
		constexpr std::array<const char*, 3> Names{ "AND" , "OR", "XOR" };

		template <typename VecLike>
		__host__ __device__ inline constexpr auto MakeInsideBox(
			std::enable_if_t<std::is_arithmetic<decltype(std::declval<VecLike>().x())>::value, VecLike> const& p_orig)
		{
			return [p_orig](VecLike const& min, VecLike const& max) {
				return !(
					p_orig.x() < min.x() || p_orig.y() < min.y() || p_orig.z() < min.z() ||
					p_orig.x() > max.x() || p_orig.y() > max.y() || p_orig.z() > max.z());
				};
		}

		template <typename VecLike>
		__host__ __device__ inline constexpr auto MakeInsideBox(
			std::enable_if_t<std::is_arithmetic<decltype(std::declval<VecLike>().x)>::value, VecLike> const& p_orig)
		{
			return [p_orig](VecLike const& min, VecLike const& max) {
				return !(
					p_orig.x < min.x || p_orig.y < min.y || p_orig.z < min.z ||
					p_orig.x > max.x || p_orig.y > max.y || p_orig.z > max.z);
				};
		}

		template<typename VecLike>
		__host__ __device__ inline constexpr auto IsCulledByBoxes(
			VecLike p_orig,
			VecLike const* const boxmin,
			VecLike const* const boxmax,
			int boxcount,
			Operator op)
		{
			if (boxcount == 0)
				return false;

			const auto inside_box = MakeInsideBox<VecLike>(p_orig);
			bool inside = inside_box(boxmin[0], boxmax[0]);
			switch (op) {
			case Operator::AND:
				for (int i = 1; i < boxcount; ++i) {
					inside &= inside_box(boxmin[i], boxmax[i]);
				}
				break;
			case Operator::OR:
				for (int i = 1; i < boxcount; ++i) {
					inside |= inside_box(boxmin[i], boxmax[i]);
				}
				break;
			case Operator::XOR:
				for (int i = 1; i < boxcount; ++i) {
					inside ^= inside_box(boxmin[i], boxmax[i]);
				}
				break;
			}
			return !inside;
		}
	}
}