#pragma once
#include <array>
#include <functional>
#include <type_traits>
#include "cuda_runtime.h"

namespace FORWARD {
	namespace Cull {

		class Operator {
		public:
			enum Value { AND, OR, XOR };
			static constexpr std::array<const char*, 3> Names{ "AND" , "OR", "XOR" };

			__host__ __device__ constexpr Operator(Value v) : v(v) {}
			__host__ __device__ inline constexpr explicit operator Value() const { return v; }

			template <typename Callable>
			__host__ __device__ inline constexpr bool Reduce(int numReductions, Callable&& next) const
			{
#define MAKE_APPLY(symbol) __host__ __device__ static inline constexpr bool apply(bool sum, bool next) { return sum symbol next; }
				struct And { MAKE_APPLY(&) };
				struct Or { MAKE_APPLY(| ) };
				struct Xor { MAKE_APPLY(^) };
#undef MAKE_APPLY

				switch (v) {
				case AND: return Reduce<Callable, And>(numReductions, std::forward<Callable&&>(next));
				case OR: return Reduce<Callable, Or>(numReductions, std::forward<Callable&&>(next));
				case XOR: return Reduce<Callable, Xor>(numReductions, std::forward<Callable&&>(next));
				}
			}

		private:
			template <typename Callable, typename Reducer>
			__host__ __device__ inline constexpr bool Reduce(int numReductions, Callable&& next) const
			{
				if (numReductions <= 0)
					return false;

				auto sum = next(0);
				for (int i = 1; i < numReductions; ++i)
					sum = Reducer::apply(sum, next(i));
				return sum;
			}

		private:
			Value v;
		};


		template <typename VecLike>
		__host__ __device__ inline static constexpr auto InsideBox(
			VecLike const& p_orig,
			VecLike const& min,
			VecLike const& max) -> std::enable_if_t<std::is_arithmetic<std::decay_t<decltype(std::declval<VecLike>().x())>>::value, bool>
		{
			return !(
				p_orig[0] < min[0] || p_orig[1] < min[1] || p_orig[2] < min[2] ||
				p_orig[0] > max[0] || p_orig[1] > max[1] || p_orig[2] > max[2]);
		}

		template <typename VecLike>
		__host__ __device__ inline static constexpr auto InsideBox(
			VecLike const& p_orig,
			VecLike const& min,
			VecLike const& max) -> std::enable_if_t<std::is_arithmetic<std::decay_t<decltype(std::declval<VecLike>().x)>>::value, bool>
		{
			return !(
				p_orig.x < min.x || p_orig.y < min.y || p_orig.z < min.z ||
				p_orig.x > max.x || p_orig.y > max.y || p_orig.z > max.z);
		} 

		template <typename VecLike>
		struct Boxes {
			VecLike const* boxmin;
			VecLike const* boxmax;
			int boxcount;

			__host__ __device__ inline constexpr bool TryCull(VecLike const& p_orig, Operator op) const
			{
				const auto inside = op.Reduce(boxcount, [=](int i) { return InsideBox(p_orig, boxmin[i], boxmax[i]); });
				return !inside;
			}
		};
	}
}

