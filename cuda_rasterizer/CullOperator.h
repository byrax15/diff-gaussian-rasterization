#pragma once
#include <array>
#include <string_view>

namespace FORWARD {
	namespace Cull {
		enum class Operator { AND, OR, XOR };
		constexpr std::array<const char*, 3> Names{ "AND" , "OR", "XOR" };
	}
}

