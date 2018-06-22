#pragma once

#include "functions.h"




namespace math
{

	/*
		仅支持浮点类型和无符号char
	*/

	template <typename T>
	class Accum
	{
	public:
		T v;
		float w;

	public:
		/** Leaves internal value uninitialized. */
		Accum(void);

		/** Initializes the internal value (usually to zero). */
		Accum(T const& init);

		/** Adds the weighted given value to the internal value. */
		void add(T const& value, float weight);

		/** Subtracts the weighted given value from the internal value. */
		void sub(T const& value, float weight);

		/**
		* Returns a normalized version of the internal value,
		* i.e. dividing the internal value by the given weight.
		* The internal value is not changed by this operation.
		*/
		T normalized(float weight) const;

		/**
		* Returns a normalized version of the internal value,
		* i.e. dividing the internal value by the internal weight,
		* which is the cumulative weight from the 'add' calls.
		*/
		T normalized(void) const;
	};

	/* ------------------------- Implementation ----------------------- */

	template <typename T>
	inline
		Accum<T>::Accum(void)
		: w(0.0f)
	{
	}

	template <typename T>
	inline
		Accum<T>::Accum(T const& init)
		: v(init), w(0.0f)
	{
	}

	template <typename T>
	inline void
		Accum<T>::add(T const& value, float weight)
	{
		this->v += value * weight;
		this->w += weight;
	}

	template <typename T>
	inline void
		Accum<T>::sub(T const& value, float weight)
	{
		this->v -= value * weight;
		this->w -= weight;
	}

	template <typename T>
	inline T
		Accum<T>::normalized(float weight) const
	{
		return this->v / weight;
	}

	template <typename T>
	inline T
		Accum<T>::normalized(void) const
	{
		return this->v / this->w;
	}

	/* ------------------------- 特例化 ----------------------- */

	template <>
	class Accum<unsigned char>
	{
	public:
		float v;
		float w;

	public:
		Accum(void);
		Accum(unsigned char const& init);
		void add(unsigned char const& value, float weight);
		void sub(unsigned char const& value, float weight);
		unsigned char normalized(float weight) const;
		unsigned char normalized(void) const;
	};

	/* ------------------------- Implementation ----------------------- */

	inline
		Accum<unsigned char>::Accum(void)
		: w(0.0f)
	{
	}

	inline
		Accum<unsigned char>::Accum(unsigned char const& init)
		: v(init), w(0.0f)
	{
	}

	inline void
		Accum<unsigned char>::add(unsigned char const& value, float weight)
	{
		this->v += static_cast<float>(value) * weight;
		this->w += weight;
	}

	inline void
		Accum<unsigned char>::sub(unsigned char const& value, float weight)
	{
		this->v -= static_cast<float>(value) * weight;
		this->w -= weight;
	}

	inline unsigned char
		Accum<unsigned char>::normalized(float weight) const
	{
		return static_cast<unsigned char>(math::round(this->v / weight));
	}

	inline unsigned char
		Accum<unsigned char>::normalized(void) const
	{
		return static_cast<unsigned char>(math::round(this->v / this->w));
	}

}
