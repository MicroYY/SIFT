#include <iostream>
#include <fstream>
#include <stdexcept>

#include "functions.h"
#include "sift.h"
#include "image_tools.h"
#include "matrix.h"
#include "matrix_tools.h"
#include "math.h"

Sift::Sift(Config const& config)
	:config(config)
{
	if (this->config.min_octave<-1 ||
		this->config.min_octave>this->config.max_octave)
		throw std::invalid_argument("Invalid octave range");

	if (this->config.contrast_threshold < 0.0f)
		this->config.contrast_threshold = 0.02f
		/ static_cast<float>(this->config.num_samples_per_octave);
	//this->config.contrast_threshold = -1;
	if (this->config.debug_output)
		this->config.verbose_output = true;
}


void
Sift::set_image(image::ByteImage::ConstPtr img)
{
	if (img->channels() != 1 && img->channels() != 3)
		throw std::invalid_argument("Gray or color image expected");

	this->orig = image::byte_to_float_image(img);
	if (img->channels() == 3)
	{
		this->orig = image::desaturate<float>
			(this->orig, image::DESATURATE_AVERAGE);
	}
}


void
Sift::set_float_image(image::FloatImage::ConstPtr img)
{
	if (img->channels() != 1 && img->channels() != 3)
		throw std::invalid_argument("Gray or color image expected");

	if (img->channels() == 3)
	{
		this->orig = image::desaturate<float>
			(img, image::DESATURATE_AVERAGE);
	}
	else
	{
		this->orig = img->duplicate();
	}
}


void
Sift::process()
{

	/*
		ͨ�������߶ȿռ�ͼ���DoG����ͼƬ�ĳ߶ȿռ���
	*/
	if (this->config.verbose_output)
	{
		std::cout << "SIFT: Creating "
			<< (this->config.max_octave - this->config.min_octave)
			<< " octaves (" << this->config.min_octave << " to "
			<< this->config.max_octave << ")..." << std::endl;
	}

	this->create_octaves();
	if (this->config.debug_output)
	{

	}


	/*
		���DoG�ֲ���ֵ
	*/
	if (this->config.debug_output)
	{
		std::cout << "SIFT: Detecting local extrema..." << std::endl;
	}

	this->extrema_detection();
	if (this->config.debug_output)
	{
		std::cout << "SIFT: Detected " << this->keypoints.size()
			<< " keypoints." << std::endl;
	}

	/*
		��ȷ�Ĺؼ��㶨λ�����
	*/
	if (this->config.debug_output)
	{
		std::cout << "SIFT: Localizing and filtering keypoints..." << std::endl;
	}
	this->keypoint_localization();
	if (this->config.debug_output)
	{
		std::cout << "SIFT: Retained " << this->keypoints.size()
			<< " stable keypoints." << std::endl;
	}

	//DoG������Ҫ
	for (std::size_t i = 0; i < this->octaves.size(); ++i)
		this->octaves[i].dog.clear();


	/*
		���ɹؼ���������
		���ܻ���ڹؼ�����������Ϊ��ÿ���ؼ�����ܻ��������������
	*/
	if (this->config.verbose_output)
	{
		std::cout << "SIFT: Generating keypoints descriptors..." << std::endl;
	}
	this->descriptor_generation();
	if (this->config.debug_output)
	{
		std::cout << "SIFT: Generated " << this->descriptors.size()
			<< " descriptors." << std::endl;
	}
	if (this->config.verbose_output)
	{
		std::cout << "SIFT: Generated " << this->descriptors.size()
			<< " descriptors from " << this->keypoints.size()
			<< " keypoints." << std::endl;
	}
	this->octaves.clear();
}



void
Sift::create_octaves()
{
	this->octaves.clear();

	/*
		���� octave -1
		��ΪԭʼͼƬģ�� sigma = 0.5, ��double sizeͼ sigma = 1
	*/
	if (this->config.min_octave < 0)
	{
		image::FloatImage::Ptr img =
			image::rescale_double_size_supersample<float>(this->orig);
		this->add_octave(img, this->config.inherent_blur_sigma * 2.0f,
			this->config.base_blur_sigma);
	}

	/*
		first positive octave ������
		���� min_octave > 0 ʱִ��
	*/
	image::FloatImage::ConstPtr img = this->orig;
	for (int i = 0; i < this->config.min_octave; ++i)
		img = image::rescale_half_size_gaussian<float>(img);


	/*
		��img������octave
		sigma * 2,Ϊ��һ��octave��ȡ�µ�base image
	*/

	float img_sigma = this->config.inherent_blur_sigma;
	for (int i = std::max(0, this->config.min_octave);
		i <= this->config.max_octave; ++i)
	{
		//TODO: �ڶ���ѭ����ʼ��������sigma����ͬ?????
		this->add_octave(img, img_sigma, this->config.base_blur_sigma);
		img = image::rescale_half_size_gaussian<float>(img);
		img_sigma = this->config.base_blur_sigma;
	}
}


void
Sift::add_octave(image::FloatImage::ConstPtr image, float has_sigma, float target_sigma)
{
	/*
		has_sigma image�Ѿ���has_sigmaƽ������
		target_sigma Ҫ���image���е�ƽ��
		��has_sigma�Ļ�����ƽ����target_sigma
		L * g(sigma1) * g(sigma2) = L * g(sqrt(sigma1^2 + sigma2^2)),
		sigma = sqrt(target_sigma^2 - has_sigma^2)
	*/
	float sigma = std::sqrt(MATH_POW2(target_sigma) - MATH_POW2(has_sigma));

	//TODO:  ���ж��ټ���sigma???

	//ƽ������ʱ����ƽ��
	//��һ��ģ����base��һ��ģ���õ�
	image::FloatImage::Ptr base = (target_sigma > has_sigma
		? image::blur_gaussian<float>(image, sigma)
		: image->duplicate());

	//�����µ�octave�������octave�ļ�����
	this->octaves.push_back(Octave());
	Octave& oct = this->octaves.back();
	//��base�����octave�ĸ�˹�ռ�
	oct.img.push_back(base);

	//  k  �߶ȿռ�֮��ĳ�������
	//  k = 2^(1/s)
	float const k = std::pow(2.0f, 1.0f / this->config.num_samples_per_octave);
	sigma = target_sigma;

	//����һ��octave����ʣ���s+2����˹�߶ȿռ�
	for (int i = 1; i < this->config.num_samples_per_octave + 3; ++i)
	{
		//Ҫ��ﵽ�ĸ�˹ģ�� sigmak = sigma * k
		float sigmak = sigma * k;
		float blur_sigma = std::sqrt(MATH_POW2(sigmak) - MATH_POW2(sigma));

		//�����µĳ߶ȿռ����� sigmak
		image::FloatImage::Ptr img = image::blur_gaussian<float>(base, blur_sigma);
		oct.img.push_back(img);

		//������˹���DoG
		image::FloatImage::Ptr dog = image::subtract<float>(img, base);
		oct.dog.push_back(dog);

		//������һ������˹ģ���Ļ���ͼ���sigma
		base = img;
		sigma = sigmak;

	}
}


void
Sift::extrema_detection()
{
	this->keypoints.clear();

	/*
		ÿһ��octave���ؼ���
	*/
	for (std::size_t i = 0; i < this->octaves.size(); ++i)
	{
		Octave const& oct(this->octaves[i]);

		//ѭ�����Σ��ҳ������߶�
		for (int s = 0; s < (int)oct.dog.size() - 2; ++s)
		{
			//ÿ��octaveȡ������DoG�����
			image::FloatImage::ConstPtr samples[3] =
			{
				// 0 1 2
				// 1 2 3
				// 2 3 4
				//�м��Ϊʵ��Ѱ�Ҽ�ֵ��ĳ߶�
				oct.dog[s + 0],
				oct.dog[s + 1],
				oct.dog[s + 2]
			};
			this->extrema_detection(samples, static_cast<int>(i)
				+ this->config.min_octave, s);
		}
	}
}


// TODO: ��ͨ��ͼƬ��ֵ��
std::size_t
Sift::extrema_detection(image::FloatImage::ConstPtr s[3], int oi, int si)
{
	// s  ����DoG
	// oi ��ǰoctave����
	// si �߶���������  ����0 1 2 ����֮һ

	int const w = s[1]->width();
	int const h = s[1]->height();

	/*       ����������Ľڵ��ƫ��       */
	int noff[9] = { -1 - w,0 - w,1 - w,-1,0,1,-1 + w,0 + w,1 + w };

	/*
		����s[1]�е��������أ�����Ƿ�������Ϊ��ֵ
	*/
	int detected = 0;
	int off = w;
	//��Ե���ֵĲ����
	for (int y = 1; y < h - 1; ++y, off += w)
		for (int x = 1; x < w - 1; ++x)
		{
			//������Ϊ w+1 �����ؿ�ʼ������һ���Ǳ�Ե����
			//   w+1 w+2 w+3... 2w-2
			//  2w+1 2w+2 2w+3...3w-2
			//  ...
			int idx = off + x;

			bool largest = true;
			bool smallest = true;
			//���ĵ�����ֵ
			float center_value = s[1]->at(idx);

			//������Χ26�������ж��Ƿ�Ϊ��ֵ
			for (int l = 0; (largest || smallest) && l < 3; ++l)
				for (int i = 0; (largest || smallest) && i < 9; ++i)
				{
					if (l == 1 && i == 4)
						continue;
					if (s[l]->at(idx + noff[i]) >= center_value)
						largest = false;
					if (s[l]->at(idx + noff[i]) <= center_value)
						smallest = false;
				}
			//�Ǽ�ֵ���ж���һ������
			if (!smallest && !largest)
				continue;

			Keypoint kp;
			kp.octave = oi;							///octave����
			kp.x = static_cast<float>(x);			///����x��λ��
			kp.y = static_cast<float>(y);			///����y��λ��
			kp.sample = static_cast<float>(si);		///�߶������������� 0 1 2����֮һ
			this->keypoints.push_back(kp);
			detected += 1;
		}
	return detected;
}


void
Sift::keypoint_localization()
{
	/*
		�������йؼ��㣬ͨ��̩��չ��ʽ��ȷ��λ��ֵ
	*/
	int num_singular = 0;
	int num_keypoints = 0;
	for (std::size_t i = 0; i < this->keypoints.size(); ++i)
	{
		//���ƹؼ���
		Keypoint kp(this->keypoints[i]);

		//��ö�Ӧ��octave ��DoG
		Octave const& oct(this->octaves[kp.octave - this->config.min_octave]);
		int sample = static_cast<int>(kp.sample);
		image::FloatImage::ConstPtr dogs[3] =
		{
			// 0 1 2
			// 1 2 3
			// 2 3 4
			oct.dog[sample + 0],oct.dog[sample + 1],oct.dog[sample + 2]
		};

		int const w = dogs[0]->width();
		int const h = dogs[0]->height();
		//�ؼ���λ�õ�int��float,���������Ϊ��ƫ��
		int ix = static_cast<int>(kp.x);
		int iy = static_cast<int>(kp.y);
		int is = static_cast<int>(kp.sample);
		float fx, fy, fs;

		//һ�׶��׵�
		float Dx, Dy, Ds;
		float Dxx, Dyy, Dss;
		float Dxy, Dxs, Dys;

		/*
			ʹ��̩��չ�������׶�λ�ؼ���
			�ؼ���ƫ���������� >0.6 ʱ��������
		*/
#define AT(S,OFF) (dogs[S]->at(px + OFF))

		//����5��
		for (int j = 0; j < 5; ++j)
		{
			//px��������
			std::size_t px = iy * w + ix;


			// Dx Dy Ds һ��ƫ��
			Dx = (AT(1, 1) - AT(1, -1)) * 0.5f;
			Dy = (AT(1, w) - AT(1, -w)) * 0.5f;
			Ds = (AT(2, 0) - AT(0, 0))  * 0.5f;

			Dxx = AT(1, 1) + AT(1, -1) - 2.0f * AT(1, 0);
			Dyy = AT(1, w) + AT(1, -w) - 2.0f * AT(1, 0);
			Dss = AT(2, 0) + AT(0, 0)  - 2.0f * AT(1, 0);

			Dxy = (AT(1, 1 + w) + AT(1, -1 - w) - AT(1, -1 + w) - AT(1, 1 - w)) * 0.25f;
			Dxs = (AT(2, 1)     + AT(0, -1)     - AT(2, -1)     - AT(0, 1))     * 0.25f;
			Dys = (AT(2, w)     + AT(0, -w)     - AT(2, -w)     - AT(0, w))     * 0.25f;

			//Hessian����A
			math::Matrix3f A;
			A[0] = Dxx; A[1] = Dxy; A[2] = Dxs;
			A[3] = Dxy; A[4] = Dyy; A[5] = Dys;
			A[6] = Dxs; A[7] = Dys; A[8] = Dss;

			//��������ʽ
			float detA = math::matrix_determinant(A);
			//�������ʽΪ0��singular matrix�������
			//�������
			if (MATH_EPSILON_EQ(detA, 0.0f, 1e-15f))
			{
				num_singular += 1;
				fx = fy = fs = 0.0f;
				break;
			}

			//�����ȷ����ȷ��
			A = math::matrix_inverse(A, detA);
			math::Vec3f b(-Dx, -Dy, -Ds);
			//hessian���������� = ������hessian���󣿣���
			b = A * b;
			fx = b[0]; fy = b[1]; fs = b[2];

			//��龫ȷλ���Ƿ�Զ����������
			int dx = (fx > 0.6f && ix < w - 2) * 1 + (fx < -0.6f && ix > 1) * -1;
			int dy = (fy > 0.6f && iy < h - 2) * 1 + (fy < -0.6f && iy > 1) * -1;

			//�����ȷλ������һ�����ظ���
			//���¶�λ
			if (dx != 0 || dy != 0)
			{
				ix += dx;
				iy += dy;
				continue;
			}

			//��ȷλ������Ҫ��
			break;
		}

		//����D(x),����̩�չ�ʽ��һ�׵�Ϊ0
		float val = dogs[1]->at(ix, iy, 0) + 0.5f * (Dx * fx + Dy * fy + Ds * fs);

		//�����Ե��Ӧ Tr(H)^2 / Det(H)
		// 2 * 2 hessian����
		//����ļ�
		float hessian_trace = Dxx + Dyy;
		//����ʽ
		float hessian_det = Dxx * Dyy - MATH_POW2(Dxy);
		//��Ե��Ӧֵ
		float hessian_score = MATH_POW2(hessian_trace) / hessian_det;
		// ��r+1)^2/r
		float score_thres = MATH_POW2(this->config.edge_ratio_threshold + 1.0f)
			/ this->config.edge_ratio_threshold;

		//�������վ�ȷ�ؼ���
		kp.x = (float)ix + fx;
		kp.y = (float)iy + fy;
		kp.sample = (float)is + fs;

		/*
			�������¹ؼ���
		 * 1. low contrast (value of DoG function at keypoint),
			�ͶԱȶ� ��˹��ֺ���ֵ

		 * 2. negative hessian determinant (curvatures with different sign),
		 *    Note that negative score implies negative determinant.
			����hessian��������ʽ

		 * 3. large edge response (large hessian score),
			����ı�Ե��Ӧ

		 * 4. unstable keypoint accurate locations,
			���ȶ��Ĺؼ��㾫ȷλ��

		 * 5. keypoints beyond the scale space boundary.
			�����߶ȿռ�߽�Ĺؼ���
		*/

		if (std::abs(val) < this->config.contrast_threshold
			|| hessian_score < 0.0f || hessian_score > score_thres
			|| std::abs(fx) > 1.5f || std::abs(fy) > 1.5f || std::abs(fs) > 1.0f
			|| kp.sample < -1.0f
			|| kp.sample >(float)this->config.num_samples_per_octave
			|| kp.x < 0.0f || kp.x >(float)(w - 1)
			|| kp.y < 0.0f || kp.y >(float)(h - 1))
		{
			//rejected
			continue;
		}

		this->keypoints[num_keypoints] = kp;
		num_keypoints += 1;
	}
	this->keypoints.resize(num_keypoints);

	if (this->config.debug_output && num_singular > 0)
	{
		std::cout << "SIFT: Warning: " << num_singular
			<< " singular matrices detected!" << std::endl;
	}
}


void
Sift::descriptor_generation()
{
	if (this->octaves.empty())
		throw std::runtime_error("Octaves not available!");
	if (this->keypoints.empty())
		return;

	this->descriptors.clear();
	this->descriptors.reserve(this->keypoints.size() * 3 / 2);

	/*
	* Keep a buffer of S+3 gradient and orientation images for the current
	* octave. Once the octave is changed, these images are recomputed.
	* To ensure efficiency, the octave index must always increase, never
	* decrease, which is enforced during the algorithm.
	*/
	int octave_index = this->keypoints[0].octave;
	Octave* octave = &this->octaves[octave_index - this->config.min_octave];
	this->generate_grad_ori_images(octave);

	/*
		�����йؼ������������
	*/
	for (std::size_t i = 0; i < this->keypoints.size(); ++i)
	{
		Keypoint const& kp(this->keypoints[i]);

		/* Generate new gradient and orientation images if octave changed. */
		if (kp.octave > octave_index)
		{
			//����ɵ�
			if (octave)
			{
				octave->grad.clear();
				octave->ori.clear();
			}
			//�����µ�gradient ��orientation
			octave_index = kp.octave;
			octave = &this->octaves[octave_index - this->config.min_octave];
			this->generate_grad_ori_images(octave);
		}
		else if (kp.octave < octave_index)
		{
			throw std::runtime_error("Decreasing octave index!");
		}


		//����ֵ
		std::vector<float> orientations;
		orientations.reserve(8);
		this->orientation_assignment(kp, octave, orientations);

		//������ȡ
		for (std::size_t j = 0; j < orientations.size(); ++j)
		{
			Descriptor desc;
			float const scale_factor = std::pow(2.0f, kp.octave);
			desc.x = scale_factor * (kp.x + 0.5f) - 0.5f;
			desc.y = scale_factor * (kp.y + 0.5f) - 0.5f;
			desc.scale = this->keypoint_absolute_scale(kp);
			desc.orientation = orientations[j];
			if (this->descriptor_assignment(kp, desc, octave))
				this->descriptors.push_back(desc);
		}
	}
}


void
Sift::generate_grad_ori_images(Octave* octave)
{
	octave->grad.clear();
	octave->grad.reserve(octave->img.size());
	octave->ori.clear();
	octave->ori.reserve(octave->img.size());

	int const width = octave->img[0]->width();
	int const height = octave->img[0]->height();

	for (std::size_t i = 0; i < octave->img.size(); ++i)
	{
		// TODO: img��ͨ��?????
		image::FloatImage::ConstPtr img = octave->img[i];
		image::FloatImage::Ptr grad = image::FloatImage::create(width, height, 1);
		image::FloatImage::Ptr ori = image::FloatImage::create(width, height, 1);

		int image_iter = width + 1;
		for (int y = 1; y < height - 1; ++y, image_iter += 2)
			for (int x = 1; x < width - 1; ++x, ++image_iter)
			{
				// dx(i,j) = [I(i+1,j) - I(i-1,j)]/2; 
				// dy(i,j) = [I(i,j+1) - I(i,j-1)]/2; 
				float m1x = img->at(image_iter - 1);
				float p1x = img->at(image_iter + 1);
				float dx = 0.5f * (p1x - m1x);

				
				float m1y = img->at(image_iter - width);
				float p1y = img->at(image_iter + width);
				float dy = 0.5f * (p1y - m1y);

				//�ݶȵ�ģֵ�ͷ��򣬷���С��0���2pi
				float atan2f = std::atan2(dy, dx);
				grad->at(image_iter) = std::sqrt(dx * dx + dy * dy);
				ori->at(image_iter) = atan2f < 0.0f
					? atan2f + MATH_PI * 2.0f : atan2f;
			}
		octave->grad.push_back(grad);
		octave->ori.push_back(ori);
	}
}


void
Sift::orientation_assignment(Keypoint const& kp,
	Octave const* octave, std::vector<float>& orientations)
{
	int const nbins = 36;
	float const nbinsf = static_cast<float>(nbins);

	//36������ֱ��ͼ
	float hist[nbins];
	std::fill(hist, hist + nbins, 0.0f);

	//�� x y sample �������룬�ҵ�����ĳ߶�����
	int const ix = static_cast<int>(kp.x + 0.5f);
	int const iy = static_cast<int>(kp.y + 0.5f);
	int const is = static_cast<int>(math::round(kp.sample));
	float const sigma = this->keypoint_relative_scale(kp);

	//TODO: why is + 1?????
	image::FloatImage::ConstPtr grad(octave->grad[is + 1]);
	image::FloatImage::ConstPtr ori(octave->ori[is + 1]);
	int const width = grad->width();
	int const height = grad->height();

	/*
	* Compute window size 'win', the full window has  2 * win + 1  pixel.
	* The factor 3 makes the window large enough such that the gaussian
	* has very little weight beyond the window. The value 1.5 is from
	* the SIFT paper. If the window goes beyond the image boundaries,
	* the keypoint is discarded.
	���򴰿ڰ뾶 3 * 1.5 * sigma
	*/
	float const sigma_factor = 1.5f;
	int win = static_cast<int>(sigma * sigma_factor * 3.0f);
	if (ix < win || ix + win >= width || iy < win || iy + win >= height)
		return;

	// max_dist�������ƽ����dxf dyf ��ȷ�����ȡ�����ƫ��
	// center ��������
	int center = iy * width + ix;
	float const dxf = kp.x - static_cast<float>(ix);
	float const dyf = kp.y - static_cast<float>(iy);
	float const max_dist = static_cast<float>(win * win) + 0.5f;


	for (int dy = -win; dy <= win; ++dy)
	{
		//������y��ƫ��
		int const yoff = dy * width;
		for (int dx = -win; dx <= win; ++dx)
		{
			float const dist = MATH_POW2(dx - dxf) + MATH_POW2(dy - dyf);
			if (dist > max_dist)
				continue;

			// gm ��ֵ gradient magnitude
			// go ���� gradient orientation 
			float gm = grad->at(center + yoff + dx);
			float go = ori->at(center + yoff + dx);
			//����ԽԶ ����ԽС
			float weight = math::gaussian_xx(dist, sigma * sigma_factor);
			// �������ĸ�����    36 * �� / 2pi  10��һ��������36����
			int bin = static_cast<int>(nbinsf * go / (2.0f * MATH_PI));
			bin = math::clamp(bin, 0, nbins - 1);
			hist[bin] += gm * weight;
		}
	}

	//ƽ��ֱ��ͼ
	for (int i = 0; i < 6; ++i)
	{
		float first = hist[0];
		float prev = hist[nbins - 1];
		for (int j = 0; j < nbins - 1; ++j)
		{
			//��ǰ������binƽ��
			float current = hist[j];
			hist[j] = (prev + current + hist[j + 1]) / 3.0f;
			prev = current;
		}
		hist[nbins - 1] = (prev + hist[nbins - 1] + first) / 3.0f;
	}

	//�ҳ�ֱ��ͼ����߷�
	float maxh = *std::max_element(hist, hist + nbins);

	//Ѱ�ҷ�ֵ��������ֵ80%�ķ�
	for (int i = 0; i < nbins; ++i)
	{
		// TODO: �Ƿ���� ����nbins?
		//ǰ��������
		float h0 = hist[(i + nbins - 1) % nbins];
		float h1 = hist[i];
		float h2 = hist[(i + 1) % nbins];

		//��Щ������Ǿֲ���ֵ
		if (h1 <= 0.8f * maxh || h1 <= h0 || h1 <= h2)
			continue;


		/*
			���β�ֵ
			f(x) = ax^2 + bx + c, f(-1) = h0, f(0) = h1, f(1) = h2
			--> a = 1/2 (h0 - 2h1 + h2), b = 1/2 (h2 - h0), c = h1.
			x = f'(x) = 2ax + b = 0 --> x = -1/2 * (h2 - h0) / (h0 - 2h1 + h2)

			x �����������i��ƫ��
			TODO why + 0.5f?????
			o����
		*/
		float x = -0.5f * (h2 - h0) / (h0 - 2.0f * h1 + h2);
		float o = 2.0f * MATH_PI * (x + (float)i + 0.5f) / nbinsf;
		orientations.push_back(o);
	}
}


bool
Sift::descriptor_assignment(Keypoint const& kp, Descriptor& desc,
	Octave const* octave)
{
	/*
		���յ�����������С PXB * PXB * OHB = 128
		4 * 4�Ĵ��� 8��������ݶ���Ϣ
	*/

	int const PXB = 4;
	int const OHB = 8;

	//���������������ƫ��
	int const ix = static_cast<int>(kp.x + 0.5f);
	float const dxf = kp.x - static_cast<float>(ix);

	int const iy = static_cast<int>(kp.y + 0.5f);
	float const dyf = kp.y - static_cast<float>(iy);

	int const is = static_cast<int>(math::round(kp.sample));
	float const sigma = this->keypoint_relative_scale(kp);

	//TODO: why is + 1?????
	image::FloatImage::ConstPtr grad(octave->grad[is + 1]);
	image::FloatImage::ConstPtr ori(octave->ori[is + 1]);
	int const width = grad->width();
	int const height = grad->height();

	desc.data.fill(0.0f);

	//�ݶȷ������ת
	float const sino = std::sin(desc.orientation);
	float const coso = std::cos(desc.orientation);

	/*
	* Compute window size.
	* Each spacial bin has an extension of 3 * sigma (sigma is the scale
	* of the keypoint). For interpolation we need another half bin at
	* both ends in each dimension. And since the window can be arbitrarily
	* rotated, we need to multiply with sqrt(2). The window size is:
	* 2W = sqrt(2) * 3 * sigma * (PXB + 1).
	*/

	float const binsize = 3.0f * sigma;
	int win = MATH_SQRT2 * binsize * (float)(PXB + 1) * 0.5f;
	//����ؼ������ⲻ�ܹ���winΪ�߳��ľ��Σ�����false
	if (ix < win || ix + win >= width || iy < win || iy + win >= height)
		return false;

	/*
	* Iterate over the window, intersected with the image region
	* from (1,1) to (w-2, h-2) since gradients/orientations are
	* not defined at the boundary pixels. Add all samples to the
	* corresponding bin.
	*/

	//�ؼ���λ������
	int const center = iy * width + ix;
	//���������ڵ���������
	for (int dy = -win; dy <= win; ++dy)
	{
		//y����ɵ�ƫ��
		int const yoff = dy * width;
		for (int dx = -win; dx <= win; ++dx)
		{
			float const mod = grad->at(center + yoff + dx);
			float const angle = ori->at(center + yoff + dx);
			float theta = angle - desc.orientation;
			if (theta < 0.0f)
				theta += 2.0f * MATH_PI;

			//������Դ��ڵ�С������
			float const winx = (float)dx - dxf;
			float const winy = (float)dy - dyf;

			/*
			* Compute normalized coordinates w.r.t. bins. The window
			* coordinates are rotated around the keypoint. The bins are
			* chosen such that 0 is the coordinate of the first bins center
			* in each dimension. In other words, (0,0,0) is the coordinate
			* of the first bin center in the three dimensional histogram.
			*/
			float binoff = (float)(PXB - 1) / 2.0f;
			//�����ڴ����е�ʵ��λ�� x y 
			float binx = (coso * winx + sino * winy) / binsize + binoff;
			float biny = (-sino * winx + coso * winy) / binsize + binoff;
			//����ʵ�ʷ���
			float bint = theta * (float)OHB / (2.0f * MATH_PI) - 0.5f;

			//����Բ�δ��ڱ���
			//�� 0.5d ��Ȩ
			float gaussian_sigma = 0.5f*(float)PXB;
			float gaussian_weight = math::gaussian_xx
			(MATH_POW2(binx - binoff) + MATH_POW2(biny - binoff),
				gaussian_sigma);
			//��Ȩģֵ
			float contrib = mod * gaussian_weight;

			/*
			* Distribute values into bins (using trilinear interpolation).
			* Each sample is inserted into 8 bins. Some of these bins may
			* not exist, because the sample is outside the keypoint window.
			*/
			//�ڽ��� x y ���к�������
			int bxi[2] = { (int)std::floor(binx),(int)std::floor(binx) + 1 };
			int byi[2] = { (int)std::floor(biny),(int)std::floor(biny) + 1 };
			//�ڽ�����������
			int bti[2] = { (int)std::floor(bint),(int)std::floor(bint) + 1 };

			//x y ���� ��������Ӱ��Ȩ�� 
			float weights[3][2] = {
				{ (float)bxi[1] - binx, 1.0f - ((float)bxi[1] - binx) },
				{ (float)byi[1] - biny, 1.0f - ((float)byi[1] - biny) },
				{ (float)bti[1] - bint, 1.0f - ((float)bti[1] - bint) }
			};

			// Wrap around orientation histogram
			if (bti[0] < 0)
				bti[0] += OHB;
			if (bti[1] >= OHB)
				bti[1] -= OHB;
			
			/* Iterate the 8 bins and add weighted contrib to each. */
			//�����ڵ� x y ���� ��8�� bin 
			//������ʵ�����Ǹõ�Ӱ���bin
			// xstride ÿһ�е�ƫ��
			// ystride 
			int const xstride = OHB;
			int const ystride = OHB * PXB;
			for (int y = 0; y < 2; ++y)
				for (int x = 0; x < 2; ++x)
					for (int t = 0; t < 2; ++t)
					{
						//��������
						if (bxi[x] < 0 || bxi[x] >= PXB
							|| byi[y] < 0 || byi[y] >= PXB)
							continue;

						//bxi[x] x������
						//byi[y] y������
						//bti[t] ���������
						//idx���ӷ�������� [0,127]
						int idx = bti[t] + bxi[x] * xstride + byi[y] * ystride;
						desc.data[idx] += contrib * weights[0][x]
							* weights[1][y] * weights[2][t];
					}

		}
	}
	/* Normalize the feature vector. */
	desc.data.normalize();

	/* Truncate descriptor values to 0.2. */
	for (int i = 0; i < PXB * PXB * OHB; ++i)
		desc.data[i] = std::min(desc.data[i], 0.2f);

	/* Normalize once again. */
	desc.data.normalize();

	return true;
}


/*
* The scale of a keypoint is: scale = sigma0 * 2^(octave + (s+1)/S).
* sigma0 is the initial blur (1.6), octave the octave index of the
* keypoint (-1, 0, 1, ...) and scale space sample s in [-1,S+1] where
* S is the amount of samples per octave. Since the initial blur 1.6
* corresponds to scale space sample -1, we add 1 to the scale index.
*/
float
Sift::keypoint_relative_scale(Keypoint const& kp)
{
	return this->config.base_blur_sigma * std::pow(2.0f,
		(kp.sample + 1.0f) / this->config.num_samples_per_octave);
}

float
Sift::keypoint_absolute_scale(Keypoint const& kp)
{
	return this->config.base_blur_sigma * std::pow(2.0f,
		kp.octave + (kp.sample + 1.0f) / this->config.num_samples_per_octave);
}