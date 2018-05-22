#pragma once


#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

#include "image_base.h"
#include "functions.h"


namespace image
{

	template<typename T> class Image;
	typedef Image<uint8_t> ByteImage;
	typedef Image<uint16_t> RawImage;
	typedef Image<char> CharImage;
	typedef Image<float> FloatImage;
	typedef Image<double> DoubleImage;
	typedef Image<int> IntImage;

	/*
		多通道图像类
		RGBRGB....
	*/
	template<typename T>
	class Image : public TypedImageBase<T>
	{
	public:
		typedef std::shared_ptr<Image<T>> Ptr;
		typedef std::shared_ptr<Image<T> const> ConstPtr;
		typedef std::vector<T> ImageData;
		typedef T ValueType;

	public:

		Image();

		Image(int width, int height, int channels);

		Image(Image<T> const& src);

		//图像的智能指针
		static Ptr create();

		static Ptr create(int width, int height, int channels);

		static Ptr create(Image<T> const& src);

		Ptr duplicate() const;

		void fill_color(T const* color);

		void add_channels(int amount, T const& value = T(0));

		void swap_channels(int c1, int c2);

		void copy_channel(int src, int dest);

		void delete_channel(int channel);

		T const& at(int index) const;

		T const& at(int index, int channel) const;

		T const& at(int x, int y, int channel) const;

		T& at(int index);

		T& at(int index, int channel);

		T& at(int x, int y, int channel);
		//单个通道线性插值
		T linear_at(float x, float y, int channel) const;
		//所有通道线性插值
		//对每个色彩通道产生一个值，结果放在px中
		void linear_at(float x, float y, T* px) const;

		T& operator[] (int index);
		T const& operator[] (int index) const;

		T const& operator() (int index) const;
		T const& operator() (int index, int channel) const;
		T const& operator() (int x, int y, int channel) const;
		T& operator() (int index);
		T& operator() (int index, int channel);
		T& operator() (int x, int y, int channel);
	};


	ImageBase::Ptr
		create_for_type(ImageType type, int width, int height, int channels);

	inline ImageBase::Ptr
		create_for_type(ImageType type, int width, int height, int channels)
	{
		switch (type)
		{
		case IMAGE_TYPE_UINT8:
			return Image<uint8_t>::create(width, height, channels);
		case IMAGE_TYPE_UINT16:
			return Image<uint16_t>::create(width, height, channels);
		case IMAGE_TYPE_UINT32:
			return Image<uint32_t>::create(width, height, channels);
		case IMAGE_TYPE_UINT64:
			return Image<uint64_t>::create(width, height, channels);
		case IMAGE_TYPE_SINT8:
			return Image<int8_t>::create(width, height, channels);
		case IMAGE_TYPE_SINT16:
			return Image<int16_t>::create(width, height, channels);
		case IMAGE_TYPE_SINT32:
			return Image<int32_t>::create(width, height, channels);
		case IMAGE_TYPE_SINT64:
			return Image<int64_t>::create(width, height, channels);
		case IMAGE_TYPE_FLOAT:
			return Image<float>::create(width, height, channels);
		case IMAGE_TYPE_DOUBLE:
			return Image<double>::create(width, height, channels);
		default:
			break;
		}
		return ImageBase::Ptr(nullptr);
	}


	template<typename T>
	inline
		Image<T>::Image()
	{
	}

	template<typename T>
	inline
		Image<T>::Image(int width, int height, int channels)
	{
		this->allocate(width, height, channels);
	}

	template<typename T>
	inline
		Image<T>::Image(Image<T> const& src)
		:TypedImageBase<T>(src)
	{
	}

	template<typename T>
	inline typename Image<T>::Ptr
		Image<T>::create()
	{
		return Ptr(new Image<T>());
	}

	template<typename T>
	inline typename Image<T>::Ptr
		Image<T>::create(int width, int height, int channels)
	{
		return Ptr(new Image<T>(width, height, channels));
	}

	template<typename T>
	inline typename Image<T>::Ptr
		Image<T>::create(Image<T> const& src)
	{
		return Ptr(new Image<T>(src));
	}


	template<typename T>
	inline typename Image<T>::Ptr
		Image<T>::duplicate() const
	{
		return Ptr(new Image<T>(*this));
	}

	template<typename T>
	inline void
		Image<T>::fill_color(T const* color)
	{
		for (T* ptr = this->begin(); ptr != this->end(); ptr += this->c)
			std::copy(color, color + this->c, ptr);
	}

	template<typename T>
	void
		Image<T>::add_channels(int num_channels, T const& value)
	{
		if (num_channels <= 0 || !this->valid())
			return;
		//RGB RGB RGB ...... RGB RGB
		std::vector<T> tmp(this->w * this->h * (this->c + num_channels));
		typename std::vector<T>::iterator dest_ptr = tmp.end();
		typename std::vector<T>::const_iterator src_ptr = this->data.end();
		const int pixels = this->get_pixel_amount();
		for (int p = 0; p < pixels; ++p)
		{
			for (int i = 0; i < num_channels; ++i)
				*(--dest_ptr) = value;
			for (int i = 0; i < this->c; ++i)
				*(--dest_ptr) = *(--src_ptr);
		}
		this->clear += num_channels;
		std::swap(this->data, tmp);
	}

	template<typename T>
	void
		Image<T>::swap_channels(int c1, int c2)
	{
		if (!this->valid() || c1 == c2 ||
			c1 >= this->channels() || c2 >= this->channels())
			return;

		T* iter1 = &this->data[0] + c1;
		T* iter2 = &this->data[0] + c2;
		int pixels = this->get_pixel_amount();
		for (int i = 0; i < pixels; ++i, iter1 += this->c, iter2 += this->c)
			std::swap(*iter1, *iter2);
	}

	template<typename T>
	void
		Image<T>::copy_channel(int src, int dest)
	{
		if (!this->valid() || src == dest)
			return;
		if (dest < 0)
		{
			dest = this->channels();
			this->add_channels(1);
		}
		T const* src_iter = &this->data[0] + src;
		T* dst_iter = &this->data[0] + dest;
		int pixels = this->get_pixel_amount();
		for (int i = 0; p < pixels; ++i, src_iter += this->c, dst_iter += this->c)
			*dst_iter = *src_iter;
	}

	template<typename T>
	void
		Image<T>::delete_channel(int chan)
	{
		if (chan < 0 || chan >= this->channels())
			return;
		typename std::vector<T>::iterator src_iter = this->data.begin();
		typename std::vector<T>::iterator dst_iter = this->data.begin();
		for (int i = 0; src_iter != this->data.end(); ++i)
		{
			if (i % this->c == chan)
				src_iter++;
			else
				*(dst_iter++) = *(src_iter++);
		}
		this->resize(this->width(), this->height(), this->channels() - 1);
	}

	template<typename T>
	inline T const&
		Image<T>::at(int index) const
	{
		return this->data[index];
	}

	template<typename T>
	inline T const&
		Image<T>::at(int index, int channel) const
	{
		int off = index * this->channels() + channel;
		return this->data[off];
	}

	template<typename T>
	inline T const&
		Image<T>::at(int x, int y, int channel) const
	{
		int off = channel + this->channels() * (x + y * this->width());
		return this->data[off];
	}

	template<typename T>
	inline T&
		Image<T>::at(int index)
	{
		return this->data[index];
	}

	template<typename T>
	inline T&
		Image<T>::at(int index, int channel)
	{
		int off = index * this->channels() + channel;
		return this->data[off];
	}

	template<typename T>
	inline T&
		Image<T>::at(int x, int y, int channel)
	{
		int off = channel + this->channels() * (x + y * this->width());
		return this->data[off];
	}


	template<typename T>
	inline T&
		Image<T>::operator[](int index)
	{
		return this->data[index];
	}

	template<typename T>
	inline T const&
		Image<T>::operator[](int index) const
	{
		return this->data[index];
	}

	template<typename T>
	inline T const&
		Image<T>::operator()(int index) const
	{
		return this->at(index);
	}

	template<typename T>
	inline T const&
		Image<T>::operator()(int index, int channel) const
	{
		return this->at(index, channel);
	}

	template<typename T>
	inline T const&
		Image<T>::operator()(int x, int y, int channel) const
	{
		return this->at(x, y, channel);
	}

	template<typename T>
	inline T&
		Image<T>::operator()(int index)
	{
		return this->at(index);
	}

	template<typename T>
	inline T&
		Image<T>::operator() (int index, int channel)
	{
		return this->at(index, channel);
	}

	template<typename T>
	inline T&
		Image<T>::operator()(int x, int y, int channel)
	{
		return this->at(x, y, channel);
	}

	template<typename T>
	T
		Image<T>::linear_at(float x, float y, int channel) const
	{
		//超过长或宽为最大值
		//小于0时为0
		//x 范围[0,w-1]
		//y 范围[0,h-1]
		x = std::max(0.0f, std::min(static_cast<float>(this->w - 1), x));
		y = std::max(0.0f, std::min(static_cast<float>(this->h - 1), y));

		//省去了x y的小数部分
		int const floor_x = static_cast<int>(x);
		int const floor_y = static_cast<int>(y);
		//再次检查是否越界
		int const floor_xp1 = std::min(floor_x + 1, this->w - 1);
		int const floor_yp1 = std::min(floor_y + 1, this->h - 1);

		//w1 w3 被省去的小数部分
		float const w1 = x - static_cast<float>(floor_x);
		float const w0 = 1.0f - w1;
		float const w3 = y - static_cast<float>(floor_y);
		float const w2 = 1.0f - w3;

		int const rowstride = this->w * this->c;
		int const row1 = floor_y * rowstride;
		int const row2 = floor_yp1 * rowstride;
		int const col1 = floor_x * this->c;
		int const col2 = floor_xp1 * this->c;

		return math::interpolate<T>(this->at(row1 + col1 + channel), this->at(row1 + col2 + channel),
			this->at(row2 + col1 + channel), this->at(row2 + col2 + channel),
			w0 * w2, w1 * w2, w0 * w3, w1 * w3);
	}

	template<typename T>
	void
		Image<T>::linear_at(float x, float y, T* px) const
	{
		x = std::max(0.0f, std::min(static_cast<float>(this->w - 1), x));
		y = std::max(0.0f, std::min(static_cast<float>(this->h - 1), y));

		int const floor_x = static_cast<int>(x);
		int const floor_y = static_cast<int>(y);
		int const floor_xp1 = std::min(floor_x + 1, this->w - 1);
		int const floor_yp1 = std::min(floor_y + 1, this->h - 1);

		float const w1 = x - static_cast<float>(floor_x);
		float const w0 = 1.0f - w1;
		float const w3 = y - static_cast<float>(floor_y);
		float const w2 = 1.0f - w3;

		int const rowstride = this->w * this->c;
		int const row1 = floor_y * rowstride;
		int const row2 = floor_yp1 * rowstride;
		int const col1 = floor_x * this->c;
		int const col2 = floor_xp1 * this->c;

		for (int cc = 0; cc < this->c; ++cc)
		{
			px[cc] = math::interpolate<T>
				(this->at(row1 + col1 + cc), this->at(row1 + col2 + cc),
					this->at(row2 + col1 + cc), this->at(row2 + col2 + cc),
					w0 * w2, w1 * w2, w0 * w3, w1 * w3);
		}
	}

}

namespace std
{
	template<class T>
	inline void
		swap(image::Image<T>& a, image::Image<T>& b)
	{
		a.swap(b);
	}
}
