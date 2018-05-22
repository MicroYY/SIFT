#pragma once


#include <algorithm>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <ostream>

#include "algo.h"

namespace math
{
	template<typename T,int N>
	class Vector;
	typedef Vector<float, 1> Vec1f;
	typedef Vector<float, 2> Vec2f;
	typedef Vector<float, 3> Vec3f;
	typedef Vector<float, 4> Vec4f;
	typedef Vector<float, 5> Vec5f;
	typedef Vector<float, 6> Vec6f;
	typedef Vector<float, 64> Vec64f;
	typedef Vector<float, 128> Vec128f;
	typedef Vector<double, 1> Vec1d;
	typedef Vector<double, 2> Vec2d;
	typedef Vector<double, 3> Vec3d;
	typedef Vector<double, 4> Vec4d;
	typedef Vector<double, 5> Vec5d;
	typedef Vector<double, 6> Vec6d;
	typedef Vector<int, 1> Vec1i;
	typedef Vector<int, 2> Vec2i;
	typedef Vector<int, 3> Vec3i;
	typedef Vector<int, 4> Vec4i;
	typedef Vector<int, 5> Vec5i;
	typedef Vector<int, 6> Vec6i;
	typedef Vector<unsigned int, 1> Vec1ui;
	typedef Vector<unsigned int, 2> Vec2ui;
	typedef Vector<unsigned int, 3> Vec3ui;
	typedef Vector<unsigned int, 4> Vec4ui;
	typedef Vector<unsigned int, 5> Vec5ui;
	typedef Vector<unsigned int, 6> Vec6ui;
	typedef Vector<char, 1> Vec1c;
	typedef Vector<char, 2> Vec2c;
	typedef Vector<char, 3> Vec3c;
	typedef Vector<char, 4> Vec4c;
	typedef Vector<char, 5> Vec5c;
	typedef Vector<char, 6> Vec6c;
	typedef Vector<unsigned char, 1> Vec1uc;
	typedef Vector<unsigned char, 2> Vec2uc;
	typedef Vector<unsigned char, 3> Vec3uc;
	typedef Vector<unsigned char, 4> Vec4uc;
	typedef Vector<unsigned char, 5> Vec5uc;
	typedef Vector<unsigned char, 6> Vec6uc;
	typedef Vector<short, 64> Vec64s;
	typedef Vector<unsigned short, 128> Vec128us;
	typedef Vector<std::size_t, 1> Vec1st;
	typedef Vector<std::size_t, 2> Vec2st;
	typedef Vector<std::size_t, 3> Vec3st;
	typedef Vector<std::size_t, 4> Vec4st;
	typedef Vector<std::size_t, 5> Vec5st;
	typedef Vector<std::size_t, 6> Vec6st;


	//任意维度和类型的向量类
	template<typename T, int N>
	class Vector
	{
	public:
		typedef T ValueType;

		static int constexpr dim = N;

		/*************** 构造函数 ***************/

		Vector();
		//使用一个指针初始化
		explicit Vector(T const* values);
		//初始化所有元素
		explicit Vector(T const& value);
		//初始化前两个元素
		Vector(T const& v1, T const& v2);
		//初始化前三个元素
		Vector(T const& v1, T const& v2, T const& v3);
		//初始化前四个元素
		Vector(T const& v1, T const& v2, T const& v3, T const& v4);

		//利用一个更小的向量和值初始化
		Vector(Vector<T, N - 1> const& other, T const& v1);

		//从相同类型的向量复制
		Vector(Vector<T, N> const& other);
		//从不同类型的向量复制
		template<typename O>
		Vector(Vector<O, N> const& other);

		//不同类型的指针初始化
		template<typename O>
		explicit Vector(O const* values);

		/*************** 管理 ***************/

		/* 给定值填充 */
		Vector<T, N>& fill(T const& value);

		/* 给定指针复制 */
		Vector<T, N>& copy(T const* values, int num = N);

		T minimum() const;

		T maximum() const;

		T sum() const;

		T abs_sum() const;

		T product() const;

		/*************** 一元操作 ***************/

		/*     ????计算长度norm*/
		T norm() const;
		//平方范数
		T square_norm() const;

		//归一化，返回引用
		Vector<T, N>& normalize();
		//归一化，返回值
		Vector<T, N> normalized() const;

		/** Component-wise absolute-value on self, returns self. */
		Vector<T, N>& abs_value(void);
		/** Returns a component-wise absolute-value copy of self. */
		Vector<T, N> abs_valued(void) const;

		/** Component-wise negation on self, returns self. */
		Vector<T, N>& negate(void);
		/** Returns a component-wise negation on copy of self. */
		Vector<T, N> negated(void) const;

		//升序排列
		Vector<T, N>& sort_asc();
		//降序排列
		Vector<T, N>& sort_desc();
		//升序
		Vector<T, N> sorted_asc() const;
		//降序
		Vector<T, N> sorted_desc() const;

		/* for_each仿函数 */
		template<typename F>
		Vector<T, N>& apply_for_each(F functor);
		template<typename F>
		Vector<T, N> applied_for_each(F functor) const;

		/*************** 二元操作 ***************/

		//点乘
		T dot(Vector<T, N> const& other) const;

		//叉乘
		Vector<T, N> cross(Vector<T, N> const& other) const;

		/** Component-wise multiplication with another vector. */
		Vector<T, N> cw_mult(Vector<T, N> const& other) const;

		/** Component-wise division with another vector. */
		Vector<T, N> cw_div(Vector<T, N> const& other) const;

		/** Component-wise similarity using epsilon checks. */
		bool is_similar(Vector<T, N> const& other, T const& epsilon) const;

		/*************** 迭代器 ***************/

		T* begin();
		T const* begin() const;
		T* end();
		T const* end() const;

		/*************** 操作符 ***************/

		/** Dereference operator to value array. */
		T* operator* (void);
		/** Const dereference operator to value array. */
		T const* operator* (void) const;

		/** Element access operator. */
		T& operator[] (int index);
		/** Const element access operator. */
		T const& operator[] (int index) const;

		/** Element access operator. */
		T& operator() (int index);
		/** Const element access operator. */
		T const& operator() (int index) const;

		/** Comparison operator. */
		bool operator== (Vector<T, N> const& rhs) const;
		/** Comparison operator. */
		bool operator!= (Vector<T, N> const& rhs) const;

		/** Assignment operator. */
		Vector<T, N>& operator= (Vector<T, N> const& rhs);
		/** Assignment operator from different type. */
		template <typename O>
		Vector<T, N>& operator= (Vector<O, N> const& rhs);

		/** Component-wise negation. */
		Vector<T, N> operator- (void) const;

		/** Self-substraction with other vector. */
		Vector<T, N>& operator-= (Vector<T, N> const& rhs);
		/** Substraction with other vector. */
		Vector<T, N> operator- (Vector<T, N> const& rhs) const;
		/** Self-addition with other vector. */
		Vector<T, N>& operator+= (Vector<T, N> const& rhs);
		/** Addition with other vector. */
		Vector<T, N> operator+ (Vector<T, N> const& rhs) const;

		/** Component-wise self-substraction with scalar. */
		Vector<T, N>& operator-= (T const& rhs);
		/** Component-wise substraction with scalar. */
		Vector<T, N> operator- (T const& rhs) const;
		/** Component-wise self-addition with scalar. */
		Vector<T, N>& operator+= (T const& rhs);
		/** Component-wise addition with scalar. */
		Vector<T, N> operator+ (T const& rhs) const;
		/** Component-wise self-multiplication with scalar. */
		Vector<T, N>& operator*= (T const& rhs);
		/** Component-wise multiplication with scalar. */
		Vector<T, N> operator* (T const& rhs) const;
		/** Component-wise self-division by scalar. */
		Vector<T, N>& operator/= (T const& rhs);
		/** Component-wise division by scalar. */
		Vector<T, N> operator/ (T const& rhs) const;

	protected:
		T v[N];


	};

	/********************** implementation **********************/

	template <typename T, int N>
	int constexpr Vector<T, N>::dim;

	template <typename T, int N>
	inline
		Vector<T, N>::Vector(void)
	{
	}

	template <typename T, int N>
	inline
		Vector<T, N>::Vector(T const* values)
	{
		std::copy(values, values + N, v);
	}

	template <typename T, int N>
	inline
		Vector<T, N>::Vector(T const& value)
	{
		fill(value);
	}

	template <typename T, int N>
	inline
		Vector<T, N>::Vector(T const& v1, T const& v2)
	{
		v[0] = v1; v[1] = v2;
	}

	template <typename T, int N>
	inline
		Vector<T, N>::Vector(T const& v1, T const& v2, T const& v3)
	{
		v[0] = v1; v[1] = v2; v[2] = v3;
	}

	template <typename T, int N>
	inline
		Vector<T, N>::Vector(T const& v1, T const& v2, T const& v3, T const& v4)
	{
		v[0] = v1; v[1] = v2; v[2] = v3; v[3] = v4;
	}

	template <typename T, int N>
	inline
		Vector<T, N>::Vector(Vector<T, N - 1> const& other, T const& v1)
	{
		std::copy(*other, *other + N - 1, v);
		v[N - 1] = v1;
	}

	template <typename T, int N>
	inline
		Vector<T, N>::Vector(Vector<T, N> const& other)
	{
		std::copy(*other, *other + N, v);
	}

	template <typename T, int N>
	template <typename O>
	inline
		Vector<T, N>::Vector(Vector<O, N> const& other)
	{
		std::copy(*other, *other + N, v);
	}

	template <typename T, int N>
	template <typename O>
	inline
		Vector<T, N>::Vector(O const* values)
	{
		std::copy(values, values + N, v);
	}

	/*************************************************/


	/*
		叉乘
	*/
	template <typename T, int N>
	inline Vector<T, N>
		cross_product(Vector<T, N> const& /*v1*/, Vector<T, N> const& /*v2*/)
	{
		throw std::invalid_argument("Only defined for 3-vectors");
	}

	/*
		三维向量叉乘
	*/
	template <typename T>
	inline Vector<T, 3>
		cross_product(Vector<T, 3> const& v1, Vector<T, 3> const& v2)
	{
		return Vector<T, 3>(v1[1] * v2[2] - v1[2] * v2[1],
			v1[2] * v2[0] - v1[0] * v2[2],
			v1[0] * v2[1] - v1[1] * v2[0]);
	}

	/*************** 管理 ***************/
	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::fill(T const& value)
	{
		std::fill(v, v + N, value);
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::copy(T const* values, int num)
	{
		std::copy(values, values + num, this->v);
		return *this;
	}

	template <typename T, int N>
	inline T
		Vector<T, N>::minimum(void) const
	{
		return *std::min_element(v, v + N);
	}

	template <typename T, int N>
	inline T
		Vector<T, N>::maximum(void) const
	{
		return *std::max_element(v, v + N);
	}

	template <typename T, int N>
	inline T
		Vector<T, N>::sum(void) const
	{
		return std::accumulate(v, v + N, T(0), std::plus<T>());
	}

	template <typename T, int N>
	inline T
		Vector<T, N>::abs_sum(void) const
	{
		return std::accumulate(v, v + N, T(0), &algo::accum_absolute_sum<T>);
	}

	template <typename T, int N>
	inline T
		Vector<T, N>::product(void) const
	{
		return std::accumulate(v, v + N, T(1), std::multiplies<T>());
	}

	/*************** 一元操作 ***************/
	template <typename T, int N>
	inline T
		Vector<T, N>::norm(void) const
	{
		return std::sqrt(square_norm());
	}

	template <typename T, int N>
	inline T
		Vector<T, N>::square_norm(void) const
	{
		return std::accumulate(v, v + N, T(0), &algo::accum_squared_sum<T>);
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::normalize(void)
	{
		std::for_each(v, v + N, algo::foreach_divide_by_const<T>(norm()));
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::normalized(void) const
	{
		return Vector<T, N>(*this).normalize();
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::abs_value(void)
	{
		std::for_each(v, v + N, &algo::foreach_absolute_value<T>);
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::abs_valued(void) const
	{
		return Vector<T, N>(*this).abs_value();
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::negate(void)
	{
		std::for_each(v, v + N, &algo::foreach_negate_value<T>);
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::negated(void) const
	{
		return Vector<T, N>(*this).negate();
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::sort_asc(void)
	{
		std::sort(v, v + N, std::less<T>());
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::sort_desc(void)
	{
		std::sort(v, v + N, std::greater<T>());
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::sorted_asc(void) const
	{
		return Vector<T, N>(*this).sort_asc();
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::sorted_desc(void) const
	{
		return Vector<T, N>(*this).sort_desc();
	}

	template <typename T, int N>
	template <typename F>
	inline Vector<T, N>&
		Vector<T, N>::apply_for_each(F functor)
	{
		std::for_each(v, v + N, functor);
		return *this;
	}

	template <typename T, int N>
	template <typename F>
	inline Vector<T, N>
		Vector<T, N>::applied_for_each(F functor) const
	{
		return Vector<T, N>(*this).apply_for_each(functor);
	}

	/*************** 二元操作 ***************/
	template <typename T, int N>
	inline T
		Vector<T, N>::dot(Vector<T, N> const& other) const
	{
		return std::inner_product(v, v + N, *other, T(0));
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::cross(Vector<T, N> const& other) const
	{
		return cross_product(*this, other);
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::cw_mult(Vector<T, N> const& other) const
	{
		Vector<T, N> ret;
		std::transform(v, v + N, other.v, ret.v, std::multiplies<T>());
		return ret;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::cw_div(Vector<T, N> const& other) const
	{
		Vector<T, N> ret;
		std::transform(v, v + N, other.v, ret.v, std::divides<T>());
		return ret;
	}

	template <typename T, int N>
	inline bool
		Vector<T, N>::is_similar(Vector<T, N> const& other, T const& eps) const
	{
		return std::equal(v, v + N, *other, algo::predicate_epsilon_equal<T>(eps));
	}

	/*************** 迭代器 ***************/

	template <typename T, int N>
	inline T*
		Vector<T, N>::begin(void)
	{
		return v;
	}

	template <typename T, int N>
	inline T const*
		Vector<T, N>::begin(void) const
	{
		return v;
	}

	template <typename T, int N>
	inline T*
		Vector<T, N>::end(void)
	{
		return v + N;
	}

	template <typename T, int N>
	inline T const*
		Vector<T, N>::end(void) const
	{
		return v + N;
	}

	/*************** 操作符 ***************/

	template <typename T, int N>
	inline T*
		Vector<T, N>::operator* (void)
	{
		return v;
	}

	template <typename T, int N>
	inline T const*
		Vector<T, N>::operator* (void) const
	{
		return v;
	}

	template <typename T, int N>
	inline T&
		Vector<T, N>::operator[] (int index)
	{
		return v[index];
	}

	template <typename T, int N>
	inline T const&
		Vector<T, N>::operator[] (int index) const
	{
		return v[index];
	}

	template <typename T, int N>
	inline T&
		Vector<T, N>::operator() (int index)
	{
		return v[index];
	}

	template <typename T, int N>
	inline T const&
		Vector<T, N>::operator() (int index) const
	{
		return v[index];
	}

	template <typename T, int N>
	inline bool
		Vector<T, N>::operator== (Vector<T, N> const& rhs) const
	{
		return std::equal(v, v + N, *rhs);
	}

	template <typename T, int N>
	inline bool
		Vector<T, N>::operator!= (Vector<T, N> const& rhs) const
	{
		return !std::equal(v, v + N, *rhs);
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::operator= (Vector<T, N> const& rhs)
	{
		std::copy(*rhs, *rhs + N, v);
		return *this;
	}

	template <typename T, int N>
	template <typename O>
	inline Vector<T, N>&
		Vector<T, N>::operator= (Vector<O, N> const& rhs)
	{
		std::copy(*rhs, *rhs + N, v);
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::operator- (void) const
	{
		return negated();
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::operator-= (Vector<T, N> const& rhs)
	{
		std::transform(v, v + N, *rhs, v, std::minus<T>());
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::operator- (Vector<T, N> const& rhs) const
	{
		return Vector<T, N>(*this) -= rhs;
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::operator+= (Vector<T, N> const& rhs)
	{
		std::transform(v, v + N, *rhs, v, std::plus<T>());
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::operator+ (Vector<T, N> const& rhs) const
	{
		return Vector<T, N>(*this) += rhs;
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::operator-= (T const& rhs)
	{
		std::for_each(v, v + N, algo::foreach_substraction_with_const<T>(rhs));
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::operator- (T const& rhs) const
	{
		return Vector<T, N>(*this) -= rhs;
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::operator+= (T const& rhs)
	{
		std::for_each(v, v + N, algo::foreach_addition_with_const<T>(rhs));
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::operator+ (T const& rhs) const
	{
		return Vector<T, N>(*this) += rhs;
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::operator*= (T const& rhs)
	{
		std::for_each(v, v + N, algo::foreach_multiply_with_const<T>(rhs));
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::operator* (T const& rhs) const
	{
		return Vector<T, N>(*this) *= rhs;
	}

	template <typename T, int N>
	inline Vector<T, N>&
		Vector<T, N>::operator/= (T const& rhs)
	{
		std::for_each(v, v + N, algo::foreach_divide_by_const<T>(rhs));
		return *this;
	}

	template <typename T, int N>
	inline Vector<T, N>
		Vector<T, N>::operator/ (T const& rhs) const
	{
		return Vector<T, N>(*this) /= rhs;
	}


	/*************** 工具 ***************/

	//标量乘法
	template <typename T, int N>
	inline Vector<T, N>
		operator* (T const& s, Vector<T, N> const& v)
	{
		return v * s;
	}

	//标量加法
	template <typename T, int N>
	inline Vector<T, N>
		operator+ (T const& s, Vector<T, N> const& v)
	{
		return v + s;
	}

	//标量减法
	template <typename T, int N>
	inline Vector<T, N>
		operator- (T const& s, Vector<T, N> const& v)
	{
		return -v + s;
	}

	/*************** 输出流 ***************/

	/*
		序列化到输出流
	*/
	template<typename T,int N>
	inline std::ostream&
		operator<<(std::ostream& os, Vector<T, N> const& v)
	{
		for (int i = 0; i < N - 1; ++i)
			os << v[i] << " ";
		os << v[N - 1];
		return os;
	}




}