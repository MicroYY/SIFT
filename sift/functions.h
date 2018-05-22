#pragma once




namespace math
{

	/**********************************
				Interpolation
	**********************************/

	//ģ��,һ��ֵ
	template<typename T>
	inline T
		interpolate(T const& v1, float w1)
	{
		return v1 * w1;
	}
	//ָ�����ͣ�һ��ֵ
	template<>
	inline unsigned char
		interpolate(unsigned char const& v1, float w1)
	{
		return (unsigned char)((float)v1 * w1 + 0.5f);
	}

	//ģ�壬����ֵ
	template<typename T>
	inline T
		interpolate(T const& v1, T const& v2, float w1, float w2)
	{
		return v1 * w1 + v2 * w2;
	}
	//ָ�����ͣ�����ֵ
	template<>
	inline unsigned char
		interpolate(unsigned char const& v1, unsigned char const& v2,
			float w1, float w2)
	{
		return (unsigned char)((float)v1 * w1 + (float)v2 * w2 + 0.5f);
	}

	//ģ�壬����ֵ
	template<typename T>
	inline T
		interpolate(T const& v1, T const& v2, T const& v3, float w1, float w2, float w3)
	{
		return v1 * w1 + v2 * w2 + v3 * w3;
	}
	//ָ�����ͣ�����ֵ
	template <>
	inline unsigned char
		interpolate(unsigned char const& v1, unsigned char const& v2,
			unsigned char const& v3, float w1, float w2, float w3)
	{
		return (unsigned char)((float)v1 * w1 + (float)v2 * w2
			+ (float)v3 * w3 + 0.5f);
	}

	//ģ�壬�ĸ�ֵ
	template <typename T>
	inline T
		interpolate(T const& v1, T const& v2, T const& v3, T const& v4,
			float w1, float w2, float w3, float w4)
	{
		return v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4;
	}

	//ָ�����ͣ��ĸ�ֵ
	template <>
	inline unsigned char
		interpolate(unsigned char const& v1, unsigned char const& v2,
			unsigned char const& v3, unsigned char const& v4,
			float w1, float w2, float w3, float w4)
	{
		return (unsigned char)((float)v1 * w1 + (float)v2 * w2
			+ (float)v3 * w3 + (float)v4 * w4 + 0.5f);
	}


	/**********************************
				   ��˹
	**********************************/

	/*
		��˹���� f(x) = exp( -1/2 * (x/sigma)^2 )
	*/
	template <typename T>
	inline T
		gaussian(T const& x, T const& sigma)
	{
		return std::exp(-((x * x) / (T(2) * sigma * sigma)));
	}

	/*
		x�Ѿ���ƽ����
		��˹���� f(x) = exp( -1/2 * xx / sigma^2 )
	*/
	template <typename T>
	inline T
		gaussian_xx(T const& xx, T const& sigma)
	{
		return std::exp(-(xx / (T(2) * sigma * sigma)));
	}


	/*     ��������     */
	template <typename T>
	inline T
		round(T const& x)
	{
		return x > T(0) ? std::floor(x + T(0.5)) : std::ceil(x - T(0.5));
	}


	/**********************************
				  Misc
	**********************************/


	template <typename T>
	T const&
		clamp(T const& v, T const& min = T(0), T const& max = T(1))
	{
		return (v < min ? min : (v > max ? max : v));
	}
}