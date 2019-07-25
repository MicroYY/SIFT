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
		通过采样尺度空间和计算DoG创建图片的尺度空间表达
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
		检测DoG局部极值
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
		精确的关键点定位与过滤
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

	//DoG不再需要
	for (std::size_t i = 0; i < this->octaves.size(); ++i)
		this->octaves[i].dog.clear();


	/*
		生成关键点描述子
		可能会大于关键点数量，因为对每个关键点可能会产生几个描述子
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
		建立 octave -1
		认为原始图片模糊 sigma = 0.5, 则double size图 sigma = 1
	*/
	if (this->config.min_octave < 0)
	{
		image::FloatImage::Ptr img =
			image::rescale_double_size_supersample<float>(this->orig);
		this->add_octave(img, this->config.inherent_blur_sigma * 2.0f,
			this->config.base_blur_sigma);
	}

	/*
		first positive octave 降采样
		仅在 min_octave > 0 时执行
	*/
	image::FloatImage::ConstPtr img = this->orig;
	for (int i = 0; i < this->config.min_octave; ++i)
		img = image::rescale_half_size_gaussian<float>(img);


	/*
		从img创建新octave
		sigma * 2,为下一层octave获取新的base image
	*/

	float img_sigma = this->config.inherent_blur_sigma;
	for (int i = std::max(0, this->config.min_octave);
		i <= this->config.max_octave; ++i)
	{
		//TODO: 第二次循环开始，后两个sigma后相同?????
		this->add_octave(img, img_sigma, this->config.base_blur_sigma);
		img = image::rescale_half_size_gaussian<float>(img);
		img_sigma = this->config.base_blur_sigma;
	}
}


void
Sift::add_octave(image::FloatImage::ConstPtr image, float has_sigma, float target_sigma)
{
	/*
		has_sigma image已经被has_sigma平滑过了
		target_sigma 要求对image进行的平滑
		在has_sigma的基础上平滑到target_sigma
		L * g(sigma1) * g(sigma2) = L * g(sqrt(sigma1^2 + sigma2^2)),
		sigma = sqrt(target_sigma^2 - has_sigma^2)
	*/
	float sigma = std::sqrt(MATH_POW2(target_sigma) - MATH_POW2(has_sigma));

	//TODO:  先判断再计算sigma???

	//平滑不够时增加平滑
	//下一次模糊由base进一步模糊得到
	image::FloatImage::Ptr base = (target_sigma > has_sigma
		? image::blur_gaussian<float>(image, sigma)
		: image->duplicate());

	//创建新的octave并存放在octave的集合中
	this->octaves.push_back(Octave());
	Octave& oct = this->octaves.back();
	//把base加入该octave的高斯空间
	oct.img.push_back(base);

	//  k  尺度空间之间的常数因子
	//  k = 2^(1/s)
	float const k = std::pow(2.0f, 1.0f / this->config.num_samples_per_octave);
	sigma = target_sigma;

	//在这一组octave创建剩余的s+2个高斯尺度空间
	for (int i = 1; i < this->config.num_samples_per_octave + 3; ++i)
	{
		//要求达到的高斯模糊 sigmak = sigma * k
		float sigmak = sigma * k;
		float blur_sigma = std::sqrt(MATH_POW2(sigmak) - MATH_POW2(sigma));

		//创建新的尺度空间样本 sigmak
		image::FloatImage::Ptr img = image::blur_gaussian<float>(base, blur_sigma);
		oct.img.push_back(img);

		//创建高斯差分DoG
		image::FloatImage::Ptr dog = image::subtract<float>(img, base);
		oct.dog.push_back(dog);

		//更新下一次做高斯模糊的基础图像和sigma
		base = img;
		sigma = sigmak;

	}
}


void
Sift::extrema_detection()
{
	this->keypoints.clear();

	/*
		每一层octave检测关键点
	*/
	for (std::size_t i = 0; i < this->octaves.size(); ++i)
	{
		Octave const& oct(this->octaves[i]);

		//循环三次，找出三个尺度
		for (int s = 0; s < (int)oct.dog.size() - 2; ++s)
		{
			//每层octave取出三张DoG并检测
			image::FloatImage::ConstPtr samples[3] =
			{
				// 0 1 2
				// 1 2 3
				// 2 3 4
				//中间的为实际寻找极值点的尺度
				oct.dog[s + 0],
				oct.dog[s + 1],
				oct.dog[s + 2]
			};
			this->extrema_detection(samples, static_cast<int>(i)
				+ this->config.min_octave, s);
		}
	}
}


// TODO: 三通道图片极值？
std::size_t
Sift::extrema_detection(image::FloatImage::ConstPtr s[3], int oi, int si)
{
	// s  三张DoG
	// oi 当前octave索引
	// si 尺度样本索引  ，即0 1 2 其中之一

	int const w = s[1]->width();
	//std::cout << w << std::endl;
	int const h = s[1]->height();

	/*       邻域相对中心节点的偏移       */
	int noff[9] = { -1 - w,0 - w,1 - w,-1,0,1,-1 + w,0 + w,1 + w };

	/*
		迭代s[1]中的所有像素，检查是否有像素为极值
	*/
	int detected = 0;
	int off = w;
	//边缘部分的不检测
	for (int y = 1; y < h - 1; ++y, off += w)
		for (int x = 1; x < w - 1; ++x)
		{
			//从索引为 w+1 的像素开始，即第一个非边缘像素
			//   w+1 w+2 w+3... 2w-2
			//  2w+1 2w+2 2w+3...3w-2
			//  ...
			int idx = off + x;

			bool largest = true;
			bool smallest = true;
			//中心点像素值
			float center_value = s[1]->at(idx);

			//遍历周围26个像素判断是否为极值
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
			//非极值，判断下一个像素
			if (!smallest && !largest)
				continue;

			Keypoint kp;
			kp.octave = oi;							///octave索引
			kp.x = static_cast<float>(x);			///像素x轴位置
			kp.y = static_cast<float>(y);			///像素y轴位置
			kp.sample = static_cast<float>(si);		///尺度样本索引，即 0 1 2其中之一
			this->keypoints.push_back(kp);
			detected += 1;
		}
	return detected;
}


void
Sift::keypoint_localization()
{
	/*
		迭代所有关键点，通过泰勒展开式精确定位极值
	*/
	int num_singular = 0;
	int num_keypoints = 0;
	for (std::size_t i = 0; i < this->keypoints.size(); ++i)
	{
		//复制关键点
		Keypoint kp(this->keypoints[i]);

		//获得对应的octave 和DoG
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
		//关键点位置的int和float,浮点可以认为是偏移
		int ix = static_cast<int>(kp.x);
		int iy = static_cast<int>(kp.y);
		int is = static_cast<int>(kp.sample);
		float fx, fy, fs;

		//一阶二阶导
		float Dx, Dy, Ds;
		float Dxx, Dyy, Dss;
		float Dxy, Dxs, Dys;

		/*
			使用泰勒展开到二阶定位关键点
			关键点偏离中心像素 >0.6 时反复迭代
		*/
#define AT(S,OFF) (dogs[S]->at(px + OFF))

		//迭代5次
		for (int j = 0; j < 5; ++j)
		{
			//px像素索引
			std::size_t px = iy * w + ix;


			// Dx Dy Ds 一阶偏导
			Dx = (AT(1, 1) - AT(1, -1)) * 0.5f;
			Dy = (AT(1, w) - AT(1, -w)) * 0.5f;
			Ds = (AT(2, 0) - AT(0, 0))  * 0.5f;

			Dxx = AT(1, 1) + AT(1, -1) - 2.0f * AT(1, 0);
			Dyy = AT(1, w) + AT(1, -w) - 2.0f * AT(1, 0);
			Dss = AT(2, 0) + AT(0, 0) - 2.0f * AT(1, 0);

			Dxy = (AT(1, 1 + w) + AT(1, -1 - w) - AT(1, -1 + w) - AT(1, 1 - w)) * 0.25f;
			Dxs = (AT(2, 1) + AT(0, -1) - AT(2, -1) - AT(0, 1))     * 0.25f;
			Dys = (AT(2, w) + AT(0, -w) - AT(2, -w) - AT(0, w))     * 0.25f;

			//Hessian矩阵A
			math::Matrix3f A;
			A[0] = Dxx; A[1] = Dxy; A[2] = Dxs;
			A[3] = Dxy; A[4] = Dyy; A[5] = Dys;
			A[6] = Dxs; A[7] = Dys; A[8] = Dss;

			//计算行列式
			float detA = math::matrix_determinant(A);
			//如果行列式为0，singular matrix奇异矩阵
			//迭代完成
			if (MATH_EPSILON_EQ(detA, 0.0f, 1e-15f))
			{
				num_singular += 1;
				fx = fy = fs = 0.0f;
				break;
			}

			//逆矩阵确定精确点
			A = math::matrix_inverse(A, detA);
			math::Vec3f b(-Dx, -Dy, -Ds);
			//hessian矩阵的逆矩阵 = 逆矩阵的hessian矩阵？？？
			b = A * b;
			fx = b[0]; fy = b[1]; fs = b[2];

			//检查精确位置是否远离像素中心
			int dx = (fx > 0.6f && ix < w - 2) * 1 + (fx < -0.6f && ix > 1) * -1;
			int dy = (fy > 0.6f && iy < h - 2) * 1 + (fy < -0.6f && iy > 1) * -1;

			//如果精确位置离另一个像素更近
			//重新定位
			if (dx != 0 || dy != 0)
			{
				ix += dx;
				iy += dy;
				continue;
			}

			//精确位置满足要求
			break;
		}

		//计算D(x),带入泰勒公式，一阶导为0
		float val = dogs[1]->at(ix, iy, 0) + 0.5f * (Dx * fx + Dy * fy + Ds * fs);

		//计算边缘响应 Tr(H)^2 / Det(H)
		// 2 * 2 hessian矩阵
		//矩阵的迹
		float hessian_trace = Dxx + Dyy;
		//行列式
		float hessian_det = Dxx * Dyy - MATH_POW2(Dxy);
		//边缘响应值
		float hessian_score = MATH_POW2(hessian_trace) / hessian_det;
		// （r+1)^2/r
		float score_thres = MATH_POW2(this->config.edge_ratio_threshold + 1.0f)
			/ this->config.edge_ratio_threshold;

		//设置最终精确关键点
		kp.x = (float)ix + fx;
		kp.y = (float)iy + fy;
		kp.sample = (float)is + fs;

		/*
			丢弃以下关键点
		 * 1. low contrast (value of DoG function at keypoint),
			低对比度 高斯差分函数值

		 * 2. negative hessian determinant (curvatures with different sign),
		 *    Note that negative score implies negative determinant.
			负的hessian矩阵行列式

		 * 3. large edge response (large hessian score),
			过大的边缘响应

		 * 4. unstable keypoint accurate locations,
			不稳定的关键点精确位置

		 * 5. keypoints beyond the scale space boundary.
			超过尺度空间边界的关键点
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

	/*for (int i = 0; i < keypoints.size(); i++)
	{
		std::cout << keypoints[i].x << " " << keypoints[i].y << " " << keypoints[i].sample << std::endl;
	}*/

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
		对所有关键点计算描述子
	*/
	for (std::size_t i = 0; i < this->keypoints.size(); ++i)
	{
		Keypoint const& kp(this->keypoints[i]);

		/* Generate new gradient and orientation images if octave changed. */
		if (kp.octave > octave_index)
		{
			//清除旧的
			if (octave)
			{
				octave->grad.clear();
				octave->ori.clear();
			}
			//设置新的gradient 和orientation
			octave_index = kp.octave;
			octave = &this->octaves[octave_index - this->config.min_octave];
			this->generate_grad_ori_images(octave);
		}
		else if (kp.octave < octave_index)
		{
			throw std::runtime_error("Decreasing octave index!");
		}


		//方向赋值
		std::vector<float> orientations;
		orientations.reserve(8);
		this->orientation_assignment(kp, octave, orientations);

		//特征提取
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
		// TODO: img单通道?????
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

				//梯度的模值和方向，方向小于0则加2pi
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

	//36个柱的直方图
	float hist[nbins];
	std::fill(hist, hist + nbins, 0.0f);

	//对 x y sample 四舍五入，找到最近的尺度样本
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
	邻域窗口半径 3 * 1.5 * sigma
	*/
	float const sigma_factor = 1.5f;
	int win = static_cast<int>(sigma * sigma_factor * 3.0f);
	if (ix < win || ix + win >= width || iy < win || iy + win >= height)
		return;

	// max_dist最大距离的平方，dxf dyf 精确点相对取整点的偏移
	// center 中心索引
	int center = iy * width + ix;
	float const dxf = kp.x - static_cast<float>(ix);
	float const dyf = kp.y - static_cast<float>(iy);
	float const max_dist = static_cast<float>(win * win) + 0.5f;


	for (int dy = -win; dy <= win; ++dy)
	{
		//索引的y轴偏移
		int const yoff = dy * width;
		for (int dx = -win; dx <= win; ++dx)
		{
			float const dist = MATH_POW2(dx - dxf) + MATH_POW2(dy - dyf);
			if (dist > max_dist)
				continue;

			// gm 幅值 gradient magnitude
			// go 方向 gradient orientation 
			float gm = grad->at(center + yoff + dx);
			float go = ori->at(center + yoff + dx);
			//距离越远 比重越小
			float weight = math::gaussian_xx(dist, sigma * sigma_factor);
			// 计算在哪个柱中    36 * θ / 2pi  10°一个柱，共36个柱
			int bin = static_cast<int>(nbinsf * go / (2.0f * MATH_PI));
			bin = math::clamp(bin, 0, nbins - 1);
			hist[bin] += gm * weight;
		}
	}

	//平滑直方图
	for (int i = 0; i < 6; ++i)
	{
		float first = hist[0];
		float prev = hist[nbins - 1];
		for (int j = 0; j < nbins - 1; ++j)
		{
			//与前后两个bin平均
			float current = hist[j];
			hist[j] = (prev + current + hist[j + 1]) / 3.0f;
			prev = current;
		}
		hist[nbins - 1] = (prev + hist[nbins - 1] + first) / 3.0f;
	}

	//找出直方图的最高峰
	float maxh = *std::max_element(hist, hist + nbins);

	//寻找峰值大于主峰值80%的峰
	for (int i = 0; i < nbins; ++i)
	{
		// TODO: 是否可以 不加nbins?
		//前后两个柱
		float h0 = hist[(i + nbins - 1) % nbins];
		float h1 = hist[i];
		float h2 = hist[(i + 1) % nbins];

		//这些峰必须是局部极值
		if (h1 <= 0.8f * maxh || h1 <= h0 || h1 <= h2)
			continue;


		/*
			二次插值
			f(x) = ax^2 + bx + c, f(-1) = h0, f(0) = h1, f(1) = h2
			--> a = 1/2 (h0 - 2h1 + h2), b = 1/2 (h2 - h0), c = h1.
			x = f'(x) = 2ax + b = 0 --> x = -1/2 * (h2 - h0) / (h0 - 2h1 + h2)

			x 看作是相对于i的偏移
			TODO why + 0.5f?????
			o方向
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
		最终的特征向量大小 PXB * PXB * OHB = 128
		4 * 4的窗口 8个方向的梯度信息
	*/

	int const PXB = 4;
	int const OHB = 8;

	//四舍五入后的坐标和偏移
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

	//梯度方向的旋转
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
	//如果关键点向外不能构成win为边长的矩形，返回false
	if (ix < win || ix + win >= width || iy < win || iy + win >= height)
		return false;

	/*
	* Iterate over the window, intersected with the image region
	* from (1,1) to (w-2, h-2) since gradients/orientations are
	* not defined at the boundary pixels. Add all samples to the
	* corresponding bin.
	*/

	//关键点位置索引
	int const center = iy * width + ix;
	//遍历窗口内的所有像素
	for (int dy = -win; dy <= win; ++dy)
	{
		//y轴造成的偏移
		int const yoff = dy * width;
		for (int dx = -win; dx <= win; ++dx)
		{
			float const mod = grad->at(center + yoff + dx);
			float const angle = ori->at(center + yoff + dx);
			float theta = angle - desc.orientation;
			if (theta < 0.0f)
				theta += 2.0f * MATH_PI;

			//计算相对窗口的小数坐标
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
			//像素在窗口中的实际位置 x y 
			float binx = (coso * winx + sino * winy) / binsize + binoff;
			//if (binx >2)
				//std::cout << binx << std::endl;
			float biny = (-sino * winx + coso * winy) / binsize + binoff;
			//像素实际方向
			float bint = theta * (float)OHB / (2.0f * MATH_PI) - 0.5f;

			//计算圆形窗口比重
			//按 0.5d 加权
			float gaussian_sigma = 0.5f*(float)PXB;
			float gaussian_weight = math::gaussian_xx
			(MATH_POW2(binx - binoff) + MATH_POW2(biny - binoff),
				gaussian_sigma);
			//加权模值
			float contrib = mod * gaussian_weight;

			/*
			* Distribute values into bins (using trilinear interpolation).
			* Each sample is inserted into 8 bins. Some of these bins may
			* not exist, because the sample is outside the keypoint window.
			*/
			//邻近的 x y 即行和列索引
			int bxi[2] = { (int)std::floor(binx),(int)std::floor(binx) + 1 };
			//std::cout << bxi[0] << " " << bxi[1] << std::endl;
			int byi[2] = { (int)std::floor(biny),(int)std::floor(biny) + 1 };
			//邻近的两个方向
			int bti[2] = { (int)std::floor(bint),(int)std::floor(bint) + 1 };

			//x y 方向 三个距离影响权重 
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
			//即相邻的 x y 方向 共8个 bin 
			//迭代的实际上是该点影响的bin
			// xstride 每一行的偏移
			// ystride 
			int const xstride = OHB;
			int const ystride = OHB * PXB;
			for (int y = 0; y < 2; ++y)
				for (int x = 0; x < 2; ++x)
					for (int t = 0; t < 2; ++t)
					{
						//超出窗口
						if (bxi[x] < 0 || bxi[x] >= PXB
							|| byi[y] < 0 || byi[y] >= PXB)
							continue;

						//bxi[x] x的索引
						//byi[y] y的索引
						//bti[t] 方向的索引
						//idx种子方向的索引 [0,127]
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