#pragma once
#include <array>
#include <type_traits>
#include "host_defines.h"

namespace FORWARD {
	namespace Cull {
		enum class Operator { AND, OR, XOR };
		constexpr std::array<const char*, 3> Names{ "AND" , "OR", "XOR" };

		template<typename TComponent>
		using vec_component = std::is_same<std::decay<TComponent>::type, float>;

#define assert_vec(VecLike) \
			static_assert(vec_component<decltype(VecLike::x)>::value \
				&& vec_component<decltype(VecLike::y)>::value \
				&& vec_component<decltype(VecLike::z)>::value, \
				"VecLike requires float x,y,z members")

		template<typename VecLike>
		__host__ __device__ inline constexpr auto MakeInsideBoxLambda(VecLike p_orig)
		{
			assert_vec(VecLike);
			return [p_orig](VecLike min, VecLike max) {
				return !(p_orig.x < min.x || p_orig.y < min.y || p_orig.z < min.z ||
					p_orig.x > max.x || p_orig.y > max.y || p_orig.z > max.z);
				};
		}

		template<typename VecLike>
		__host__ __device__ inline constexpr auto IsCulledByBoxes(
			VecLike const* const boxmin, VecLike const* const boxmax, int boxcount, 
			Operator op, decltype(MakeInsideBoxLambda<VecLike>(VecLike{})) inside_box)
		{
			assert_vec(VecLike);

			if (boxcount == 0) 
				return false;

			bool inside = inside_box(boxmin[0], boxmax[0]);
			switch (op) {
			case Operator::AND:
				for (int i = 0; i < boxcount; ++i) {
					inside &= inside_box(boxmin[i], boxmax[i]);
				}
				break;
			case Operator::OR:
				for (int i = 0; i < boxcount; ++i) {
					inside |= inside_box(boxmin[i], boxmax[i]);
				}
				break;
			case Operator::XOR:
				for (int i = 0; i < boxcount; ++i) {
					inside ^= inside_box(boxmin[i], boxmax[i]);
				}
				break;
			}
			return !inside;
		}
	}
}

