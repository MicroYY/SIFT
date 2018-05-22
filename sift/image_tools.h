#pragma once


#include <iostream>
#include <limits>
#include <complex>
#include <type_traits>

#include <cmath>
#include <stdexcept>

#include "image.h"
#include "functions.h"
#include "accum.h"
#include "math.h"

namespace image
{

	/*
	***************ͼ��ת��***************
	*/

	/*
		byte -> float
		[0, 255] -> [0, 1]
	*/
	FloatImage::Ptr
		byte_to_float_image(ByteImage::ConstPtr image);

	/*
		byte -> double
		[0, 255] -> [0, 1]
	*/
	DoubleImage::Ptr
		byte_to_double_image(ByteImage::ConstPtr image);

	/*
		float -> byte
		[vmin, vmax] -> [0, 255]
	*/
	ByteImage::Ptr
		float_to_byte_image(FloatImage::ConstPtr image,
			float vmin = 0.0f, float vmax = 1.0f);

	/*
		double -> byte
		[vmin, vmax] -> [0, 255]
	*/
	ByteImage::Ptr
		double_to_byte_image(DoubleImage::ConstPtr image,
			double vmin = 0.0f, double vmax = 1.0f);

	/*
		int -> byte
		clamping absolute values
	*/
	ByteImage::Ptr
		int_to_byte_image(IntImage::ConstPtr image);

	/*
		raw -> byte
		[vmin, vmax] -> [0, 255]
	*/
	ByteImage::Ptr
		raw_to_byte_image(RawImage::ConstPtr image,
			uint16_t vmin = 0, uint16_t vmax = 65535);

	/*
		raw -> float
		[0, 65535] -> [0, 1]
	*/
	FloatImage::Ptr
		raw_to_float_image(RawImage::ConstPtr image);

	/*
		ģ��ת����û��scaling��clamping
	*/
	template<typename SRC, typename DST>
	typename Image<DST>::Ptr
		type_to_type_image(typename Image<SRC>::ConstPtr image);

	/*
		Ѱ����Сֵ�����ֵ
	*/
	template<typename T>
	void
		find_min_max_value(typename image::Image<T>::ConstPtr image, T* vmin, T* vmax);

	/*
		��һ��
	*/
	void
		float_image_normalize(FloatImage::Ptr image);

	/*
	***************ͼ��ȥ����***************
	*/

	/*
		������������ȥ����
		����ֵ���ʱȥ����û������
		����ģ�� Microsoft Photosynther
		����������
	*/
	template<typename T>
	typename Image<T>::Ptr
		image_undistort_msps(typename Image<T>::ConstPtr img, double k0, double k1);

	/*
		�������������ȥ�������
		����ģ��  rd(r) = 1 + k_2 r^2 + k_4 r^4 ���� r^2 = x^2 + y^2
		���� uint ����
		MVE bundler
	*/
	template<typename T>
	typename Image<T>::Ptr
		image_undistort_k2k4(typename Image<T>::ConstPtr img,
			double focal_length, double k2, double k4);

	/*
		���������һ������ȥ����
		�������Ϊ0��ȥ����û������
		���� uint ����
		VisualSfM
	*/
	template<typename T>
	typename Image<T>::Ptr
		image_undistort_vsfm(typename Image<T>::ConstPtr img,
			double focal_length, double k1);

	/*
	***************ͼ��������ü�***************
	*/

	/*
		�ü���һ����������
		Խ�������ø�������ɫ��ʼ��
	*/
	template<typename T>
	typename Image<T>::Ptr
		crop(typename Image<T>::ConstPtr image, int width, int height,
			int left, int top, T const* fill_color);

	/****** ���Ų�ֵ���� ******/
	enum RescaleInterpolation
	{
		RESCALE_NEAREST,
		RESCALE_LINEAR,
		RESCALE_GAUSSIAN  //byte image ������
	};

	/*
		�������ŵ�ͼƬ
	*/
	template<typename T>
	typename Image<T>::Ptr
		rescale(typename Image<T>::ConstPtr image, RescaleInterpolation interp,
			int width, int height);

	/*
		���ų߶�0.5
		2*2 pixels -> 1 pixel
		�ߴ�Ϊ����ʱ���µĳߴ� new_size = (old_size + 1) / 2
	*/
	template<typename T>
	typename Image<T>::Ptr
		rescale_half_size(typename Image<T>::ConstPtr image);

	/*
		��������Ϊ0.5�ĸ�˹����ͼ��
		�� 4*4, sigma ���ܱ�1��̫��
		Ĭ�� sigma = sqrt(1.0^2 - 0.5^2) =~ 0.866
	*/
	template<typename T>
	typename Image<T>::Ptr
		rescale_half_size_gaussian(typename Image<T>::ConstPtr image,
			float sigma = 0.866025403784439f);

	/*
		ÿ�������� subsample
		��ԭʼͼƬ�Ѿ��д�ŵ�ģ��ʱ����
	*/
	template<typename T>
	typename Image<T>::Ptr
		rescale_half_size_subsample(typename Image<T>::ConstPtr image);

	/*
		upscale ���Բ�ֵ ����Ϊ2
		ֻ�в�ֵֵ����������ͼ
		�����һЩԭʼ��Ϣ�������� dimension/appearance
	*/
	template<typename T>
	typename Image<T>::Ptr
		rescale_double_size(typename Image<T>::ConstPtr img);

	/*
		upscale ���Բ�ֵ
		ֱ�ӻ�ȡÿ���к���
		������������Ϣ��ԭʼͼƬ����ͨ��rescale_half_size_subsample�ָ���
		���Ͻǲ���������ص�ƫ��
	*/
	template<typename T>
	typename Image<T>::Ptr
		rescale_double_size_supersample(typename Image<T>::ConstPtr img);

	/*
		������ڵ���������ͼ��in
		��ͼ�����ŵ�out��ά�ȣ���������out��
		downsampleʱ������in��Ҫ�ں��ʵ� mipmap level����ֹʧ��
	*/
	template<typename T>
	void
		rescale_nearest(typename Image<T>::ConstPtr in, typename Image<T>::Ptr out);

	/*
		���Բ�ֵ��������in
		��ͼ�����ŵ�out��ά�ȣ���������out��
		downsampleʱ������in��Ҫ�ں��ʵ� mipmap level����ֹʧ��
	*/
	template<typename T>
	void
		rescale_linear(typename Image<T>::ConstPtr in, typename Image<T>::Ptr out);

	/*
		��˹����ģ��������in
		��ͼ�����ŵ�out��ά�ȣ���������out��
		sigmaԽС�����Խ������ʧ�������
		sigmaԽ�󣬽��Խƽ����Խģ��
		�÷����ٶȷǳ���
	*/
	template<typename T>
	void
		rescale_gaussian(typename Image<T>::ConstPtr in,
			typename Image<T>::Ptr out, float sigma_factor = 1.0f);

	/*
	***************ͼ��ģ��***************
	*/

	/*
		��˹�����ģ��
	*/
	template<typename T>
	typename Image<T>::Ptr
		blur_gaussian(typename Image<T>::ConstPtr in, float sigma);

	/*
		�˲��� size 'ks' ģ��
		�ٶȱȸ�˹ģ����
		���������˹ģ�����˲���ʧ�棩
	*/
	template<typename T>
	typename Image<T>::Ptr
		blur_boxfilter(typename Image<T>::ConstPtr in, int ks);

	/*
	***************ͼ����ת�ͷ�ת***************
	*/

	/****** ͼ����ת���� ******/
	enum RotateType
	{
		ROTATE_CCW,		///< ��ʱ��
		ROTATE_CW,		///< ˳ʱ��
		ROTATE_180,		///< 180����ת
		ROTATE_SWAP		///< ����x���y��
	};

	/*
		ʵ���κ�һ����ת
		���������Ტ������������ת�����Ǳ���
	*/
	template<typename T>
	typename Image<T>::Ptr
		rotate(typename Image<T>::ConstPtr image, RotateType type);

	/*
		�����Ƕ�˳ʱ����ת
		�ø�������ɫ���
	*/
	template<typename T>
	typename Image<T>::Ptr
		rotate(typename Image<T>::ConstPtr image, float angle, T const* fill_color);

	/****** ͼ��ת���� ******/
	enum FlipType
	{
		FLIP_NONE = 0,
		FLIP_HORIZONTAL = 1 << 0,
		FLIP_VERTICAL = 1 << 1,
		FLIP_BOTH = FLIP_HORIZONTAL | FLIP_VERTICAL
	};

	/*
		��ת����ͼƬ
	*/
	template<typename T>
	void
		flip(typename Image<T>::Ptr image, FlipType type);

	/*
	***************ͼ��ȥ����***************
	*/

	enum DesaturateType
	{
		DESATURATE_MAXIMUM,		///< max(R,G,B)
		DESATURATE_LIGHTNESS,	///< (max(R,G,B) + min(R,G,B)) * 1/2
		DESATURATE_LUMINOSITY,	///< 0.21 * R + 0.72 * G + 0.07 * B
		DESATURATE_LUMINANCE,	///< 0.30 * R + 0.59 * G + 0.11 * B
		DESATURATE_AVERAGE		///< (R + G + B) * 1/3
	};


	/**
	* Desaturates an RGB or RGBA image to G or GA respectively.
	* A new image is returned, the original image is untouched.
	*
	* From http://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
	*
	* Maximum = max(R,G,B)
	* Lightness = 1/2 * (max(R,G,B) + min(R,G,B))
	* Luminosity = 0.21 * R + 0.72 * G + 0.07 * B
	* Luminance = 0.30 * R + 0.59 * G + 0.11 * B
	* Average Brightness = 1/3 * (R + G + B)
	*/
	template<typename T>
	typename Image<T>::Ptr
		desaturate(typename Image<T>::ConstPtr image, DesaturateType type);


	/*
		�Ҷ�ͼ������˫ͨ���� -> RGB �� RGBA
	*/
	template<typename T>
	typename Image<T>::Ptr
		expand_grayscale(typename Image<T>::ConstPtr image);

	/*
		ȥalphaͨ��
	*/
	template<typename T>
	void
		reduce_alpha(typename Image<T>::Ptr img);


	/*
	***************��Ե���***************
	*/

	/*
		sobel����ʵ��
		http://en.wikipedia.org/wiki/Sobel_operator
		For byte images, the operation can lead to clipped values.
		Likewise for floating point images, it leads to values >1.
	*/
	template<typename T>
	typename Image<T>::Ptr
		sobel_edge(typename Image<T>::ConstPtr img);


	/*
	***************����***************
	*/

	/*
		ͼ��������з���
		���޷���ͼ��������Ч
	*/
	template<typename T>
	typename Image<T>::Ptr
		subtract(typename Image<T>::ConstPtr i1, typename Image<T>::ConstPtr i2);

	/*
		������ֵͼ�񣬼���ÿ��ֵ�ľ���ֵ��
		���޷���ͼ������
	*/
	template<typename T>
	typename Image<T>::Ptr
		difference(typename Image<T>::ConstPtr i1, typename Image<T>::ConstPtr i2);

	/**
	* Applies gamma correction to float/double images (in-place).
	* To obtain color values from linear intensities, use 1/2.2 as exponent.
	* To remove gamma correction from an image, use 2.2 as exponent.
	*/
	template<typename T>
	void
		gamma_correct(typename Image<T>::Ptr image, T const& power);

	/**
	* Applies fast gamma correction to byte image using a lookup table.
	* Note that alpha channels are not treated as such and are also corrected!
	*/
	void
		gamma_correct(ByteImage::Ptr image, float power);

	/**
	* Applies gamma correction to float/double images (in-place) with linear
	* RGB values in range [0, 1] to nonlinear R'G'B' values according to
	* http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html:
	*
	*   X' = 12.92 * X                   if X <= 0.0031308
	*   X' = 1.055 * X^(1/2.4) - 0.055   otherwise
	*
	* TODO: Implement overloading for integer image types.
	*/
	template <typename T>
	void
		gamma_correct_srgb(typename Image<T>::Ptr image);

	/**
	* Applies inverse gamma correction to float/double (in-place) images with
	* nonlinear R'G'B' values in the range [0, 1] to linear sRGB values according
	* to http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html:
	*
	*   X = X' / 12.92                     if X' <= 0.04045
	*   X = ((X' + 0.055) / (1.055))^2.4   otherwise
	*
	* TODO: Implement overloading for integer image types.
	*/
	template <typename T>
	void
		gamma_correct_inv_srgb(typename Image<T>::Ptr image);

	/**
	* Calculates the integral image (or summed area table) for the input image.
	* The integral image is computed channel-wise, i.e. the output image has
	* the same amount of channels as the input image.
	*/
	template <typename T_IN, typename T_OUT>
	typename Image<T_OUT>::Ptr
		integral_image(typename Image<T_IN>::ConstPtr image);

	/**
	* Sums over the rectangle defined by A=(x1,y1) and B=(x2,y2) on the given
	* SAT for channel cc. This is efficiently calculated as B + A - C - D
	* where C and D are the other two points of the rectange. The points
	* A and B are considered to be INSIDE the rectangle.
	*/
	template <typename T>
	T
		integral_image_area(typename Image<T>::ConstPtr sat,
			int x1, int y1, int x2, int y2, int cc = 0);

	/**
	* Creates a thumbnail of the given size by first rescaling the image
	* and then cropping to fill the thumbnail.
	*/
	template <typename T>
	typename Image<T>::Ptr
		create_thumbnail(typename Image<T>::ConstPtr image,
			int thumb_width, int thumb_height);





	/************************* implementation *************************/

	template <typename T>
	inline T
		desaturate_maximum(T const* v)
	{
		return *std::max_element(v, v + 3);
	}

	template <typename T>
	inline T
		desaturate_lightness(T const* v)
	{
		T const* max = std::max_element(v, v + 3);
		T const* min = std::min_element(v, v + 3);
		return math::interpolate(*max, *min, 0.5f, 0.5f);
	}

	template <typename T>
	inline T
		desaturate_luminosity(T const* v)
	{
		return math::interpolate(v[0], v[1], v[2], 0.21f, 0.72f, 0.07f);
	}

	template <typename T>
	inline T
		desaturate_luminance(T const* v)
	{
		return math::interpolate(v[0], v[1], v[2], 0.30f, 0.59f, 0.11f);
	}

	template <typename T>
	inline T
		desaturate_average(T const* v)
	{
		float third(1.0f / 3.0f);
		return math::interpolate(v[0], v[1], v[2], third, third, third);
	}


	template<typename T>
	typename Image<T>::Ptr
		desaturate(typename Image<T>::ConstPtr img, DesaturateType type)
	{
		if (img == nullptr)
			throw std::invalid_argument("Null image given");

		int ic = img->channels();
		if (ic != 3 && ic != 4)
			throw std::invalid_argument("Image must be RGB or RGBA");

		bool has_alpha = (ic == 4);

		typename Image<T>::Ptr out(Image<T>::create());
		out->allocate(img->width(), img->height(), 1 + has_alpha);

		typedef T(*DesaturateFunc)(T const*);
		DesaturateFunc func;
		switch (type)
		{
		case DESATURATE_MAXIMUM: func = desaturate_maximum<T>; break;
		case DESATURATE_LIGHTNESS: func = desaturate_lightness<T>; break;
		case DESATURATE_LUMINOSITY: func = desaturate_luminosity<T>; break;
		case DESATURATE_LUMINANCE: func = desaturate_luminance<T>; break;
		case DESATURATE_AVERAGE: func = desaturate_average<T>; break;
		default: throw std::invalid_argument("Invalid desaturate type");
		}

		int outpos = 0;
		int inpos = 0;
		int pixels = img->get_pixel_amount();
		for (int i = 0; i < pixels; ++i)
		{
			T const* v = &img->at(inpos);
			out->at(outpos) = func(v);

			if (has_alpha)
				out->at(outpos + 1) = img->at(inpos + 3);

			outpos += 1 + has_alpha;
			inpos += 3 + has_alpha;
		}

		return out;
	}


	template<typename T>
	typename Image<T>::Ptr
		rescale_double_size_supersample(typename Image<T>::ConstPtr img)
	{
		int const iw = img->width();
		int const ih = img->height();
		int const ic = img->channels();
		int const ow = iw << 1;
		int const oh = ih << 1;

		typename Image<T>::Ptr out(Image<T>::create());
		out->allocate(ow, oh, ic);

		int witer = 0;
		//��һ��������չ�� 2*2��
		for (int y = 0; y < oh; ++y)
		{
			//���һ�б��Ϊfalse
			bool nexty = (y + 1 < oh);
			// y����ƫ��
			int yoff[2] = { iw * (y >> 1), iw * ((y + nexty) >> 1) };
			for (int x = 0; x < ow; ++x)
			{
				//���һ�б��Ϊfalse
				bool nextx = (x + 1 < ow);
				// x����ƫ��
				int xoff[2] = { x >> 1,(x + nextx) >> 1 };
				//һ�����ر���չ�� 2*2 
				T const* val[4] =
				{
					//ȡ����չ�������еĵ�һ��
					&img->at(yoff[0] + xoff[0],0),
					&img->at(yoff[0] + xoff[1],0),
					&img->at(yoff[1] + xoff[0],0),
					&img->at(yoff[1] + xoff[1],0) };

				for (int c = 0; c < ic; ++c, ++witer)
					out->at(witer) = math::interpolate
					(val[0][c], val[1][c], val[2][c], val[3][c],
						0.25f, 0.25f, 0.25f, 0.25f);
			}
		}
		return out;
	}


	template<typename T>
	typename Image<T>::Ptr
		blur_gaussian(typename Image<T>::ConstPtr in, float sigma)
	{
		if (in == nullptr)
			throw std::invalid_argument("Null image given");

		//sigma̫Сû�б仯
		if (MATH_EPSILON_EQ(sigma, 0.0f, 0.1f))
			return in->duplicate();

		int const w = in->width();
		int const h = in->height();
		int const c = in->channels();
		int const ks = std::ceil(sigma * 2.884f);
		std::vector<float> kernel(ks + 1);

		//����
		for (int i = 0; i < ks + 1; ++i)
			kernel[i] = math::gaussian((float)i, sigma);

		//����һά���

		//x����
		typename Image<T>::Ptr sep(Image<T>::create(w, h, c));
		int px = 0;
		for (int y = 0; y < h; ++y)
			for (int x = 0; x < w; ++x, ++px)
				for (int cc = 0; cc < (int)c; ++cc)
				{
					//ÿ��ͨ�����������
					math::Accum<T> accum(T(0));
					for (int i = -ks; i <= ks; ++i)
					{
						int idx = math::clamp(x + i, 0, (int)w - 1);
						accum.add(in->at(y * w + idx, cc), kernel[std::abs(i)]);
					}
					sep->at(px, cc) = accum.normalized();
				}

		typename Image<T>::Ptr out(Image<T>::create(w, h, c));
		px = 0;
		for (int y = 0; y < h; ++y)
			for (int x = 0; x < w; ++x, ++px)
				for (int cc = 0; cc < c; ++cc)
				{
					math::Accum<T> accum(T(0));
					for (int i = -ks; i <= ks; ++i)
					{
						int idx = math::clamp(y + i, 0, (int)h - 1);
						accum.add(sep->at(idx * w + x, cc), kernel[std::abs(i)]);
					}
					out->at(px, cc) = accum.normalized();
				}
		return out;
	}


	template <typename T>
	typename Image<T>::Ptr
		rescale_half_size(typename Image<T>::ConstPtr img)
	{
		int const iw = img->width();
		int const ih = img->height();
		int const ic = img->channels();
		int const ow = (iw + 1) >> 1;
		int const oh = (ih + 1) >> 1;

		if (iw < 2 || ih < 2)
			throw std::invalid_argument("Input image too small for half-sizing");

		typename Image<T>::Ptr out(Image<T>::create());
		out->allocate(ow, oh, ic);

		int outpos = 0;
		int rowstride = iw * ic;
		for (int y = 0; y < oh; ++y)
		{
			int irow1 = y * 2 * rowstride;
			int irow2 = irow1 + rowstride * (y * 2 + 1 < ih);

			for (int x = 0; x < ow; ++x)
			{
				int ipix1 = irow1 + x * 2 * ic;
				int ipix2 = irow2 + x * 2 * ic;
				int hasnext = (x * 2 + 1 < iw);

				for (int c = 0; c < ic; ++c)
					out->at(outpos++) = math::interpolate<T>(
						img->at(ipix1 + c), img->at(ipix1 + ic * hasnext + c),
						img->at(ipix2 + c), img->at(ipix2 + ic * hasnext + c),
						0.25f, 0.25f, 0.25f, 0.25f);
			}
		}

		return out;
	}

	template<typename T>
	typename Image<T>::Ptr
		rescale_half_size_gaussian(typename Image<T>::ConstPtr img, float sigma)
	{
		int const iw = img->width();
		int const ih = img->height();
		int const ic = img->channels();
		int const ow = (iw + 1) >> 1;
		int const oh = (ih + 1) >> 1;

		if (iw < 2 || ih < 2)
			throw std::invalid_argument("Invalid input image");

		typename Image<T>::Ptr out(Image<T>::create());
		out->allocate(ow, oh, ic);

		/*
		* Weights w1 (4 center px), w2 (8 skewed px) and w3 (4 corner px).
		* Weights can be normalized by dividing with (4*w1 + 8*w2 + 4*w3).
		* Since the accumulator is used, normalization is not explicit.
		*/
		float const w1 = std::exp(-0.5f / (2.0f * MATH_POW2(sigma)));
		float const w2 = std::exp(-2.5f / (2.0f * MATH_POW2(sigma)));
		float const w3 = std::exp(-4.5f / (2.0f * MATH_POW2(sigma)));

		int outpos = 0;
		int const rowstride = iw * ic;
		for (int y = 0; y < oh; ++y)
		{
			/* Init the four row pointers. */
			int y2 = (int)y << 1;
			T const* row[4];
			row[0] = &img->at(std::max(0, y2 - 1) * rowstride);
			row[1] = &img->at(y2 * rowstride);
			row[2] = &img->at(std::min((int)ih - 1, y2 + 1) * rowstride);
			row[3] = &img->at(std::min((int)ih - 1, y2 + 2) * rowstride);

			for (int x = 0; x < ow; ++x)
			{
				/* Init four pixel positions for each row. */
				int x2 = (int)x << 1;
				int xi[4];
				xi[0] = std::max(0, x2 - 1) * ic;
				xi[1] = x2 * ic;
				xi[2] = std::min((int)iw - 1, x2 + 1) * ic;
				xi[3] = std::min((int)iw - 1, x2 + 2) * ic;

				/* Accumulate 16 values in each channel. */
				for (int c = 0; c < ic; ++c)
				{
					math::Accum<T> accum(T(0));
					accum.add(row[0][xi[0] + c], w3);
					accum.add(row[0][xi[1] + c], w2);
					accum.add(row[0][xi[2] + c], w2);
					accum.add(row[0][xi[3] + c], w3);

					accum.add(row[1][xi[0] + c], w2);
					accum.add(row[1][xi[1] + c], w1);
					accum.add(row[1][xi[2] + c], w1);
					accum.add(row[1][xi[3] + c], w2);

					accum.add(row[2][xi[0] + c], w2);
					accum.add(row[2][xi[1] + c], w1);
					accum.add(row[2][xi[2] + c], w1);
					accum.add(row[2][xi[3] + c], w2);

					accum.add(row[3][xi[0] + c], w3);
					accum.add(row[3][xi[1] + c], w2);
					accum.add(row[3][xi[2] + c], w2);
					accum.add(row[3][xi[3] + c], w3);

					out->at(outpos++) = accum.normalized();
				}
			}
		}

		return out;
	}



	template<typename T>
	typename Image<T>::Ptr
		subtract(typename Image<T>::ConstPtr i1, typename Image<T>::ConstPtr i2)
	{
		int const iw = i1->width();
		int const ih = i1->height();
		int const ic = i1->channels();

		if (i1 == nullptr || i2 == nullptr)
			throw std::invalid_argument("Null image given");

		if (iw != i2->width() || ih != i2->height() || ic != i2->channels())
			throw std::invalid_argument("Image dimensions do not match");


		typename Image<T>::Ptr out(Image<T>::create());
		out->allocate(iw, ih, ic);

		//TODO
		for (int i = 0; i < i1->get_value_amount(); ++i)
			out->at(i) = i1->at(i) - i2->at(i);

		return out;
	}

}