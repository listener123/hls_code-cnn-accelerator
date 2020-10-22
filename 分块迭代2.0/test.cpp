#include "top.h"
#include <iostream>
#include <fstream>

void load_data(const char* path, char* ptr, unsigned int size)
{
	std::ifstream f(path, std::ios::in | std::ios::binary);
	if (!f)
	{
		std::cout << path<<" no such file,please check the file name!/n";
		exit(0);
	}

	f.read(ptr, size);
	f.close();
}

void padding_no(float *in, float *out,unsigned f_W,unsigned f_H,unsigned f_in_ch,unsigned value)
{
	//in[batch*f_in_ch*f_W*f_H]

	//out[batch*f_in_ch*out_w*out_h]
	int p = (3 - 1) / 2;//    p=((s-1)x-s+k)/2
	int out_w = f_W + 2 * p;
	int out_h = f_H + 2 * p;

	for (int ch = 0; ch < f_in_ch; ch++)
	{
		padding_label4:for (int i = 0; i < out_w; i++)
		{
			padding_label5:for (int j=0;j<p;j++)
			{
				out[ch * out_w * out_h+i* out_h+j] = value;
			}

			padding_label6:for (int j = p; j < out_h - p; j++)
			{
				if (i < p || i >=out_w - p) out[ch * out_w * out_h + i * out_h + j] = value;
				else out[ch * out_w * out_h + i * out_h + j] = in[ch * f_W * f_H + (i - p) * f_H + j - p];
			}

			padding_label7:for (int j = out_h-p; j < out_h; j++)
			{
				out[ch * out_w * out_h + i * out_h + j] = value;
			}

		}
	}

}
void pool_no(float* in, float* out, int f_W,int f_H,int w_k,int f_in_ch,int type)
{
	//in[batch*f_in_ch*f_W*f_H]

	//out[batch*f_in_ch*out_w*out_h]
	//type0 max type1 average
	//compute
	int out_w = (f_W - w_k) / 2 + 1;
	int out_h = (f_H - w_k) / 2 + 1;

	for (int batch = 0; batch < BATCH; batch++)
	{
		for (int row = 0; row < out_w; row += 1)
		{
			for (int col = 0; col < out_h; col += 1)
			{

				for (int ti = 0; ti < f_in_ch; ti += 1)
					{

						for (int i = 0; i < w_k; i++)
						{
							for (int j = 0; j < w_k; j++)
							{
								for (int trr = row; trr < row + 1; trr++)
								{
									for (int tcc = col; tcc < col + 1; tcc++)
									{

										for (int tii = ti; tii < ti + 1; tii++)
											{
												if (type == 1) //average pool
												{
													out[batch * out_h * out_w * f_in_ch + tii * out_h * out_w + trr * out_h + tcc] +=
														in[batch * f_W * f_H * f_in_ch + tii * f_H * f_W + (2 * trr + i) * f_H + 2 * tcc + j] / (w_k * w_k);
												}
												if (type == 0) //max pool
												{
													float a = out[batch * out_h * out_w * f_in_ch + tii * out_h * out_w + trr * out_h + tcc];
													float b = in[batch * f_H * f_H * f_in_ch + tii * f_H * f_W + (2 * trr + i) * f_H + 2 * tcc + j];
													if (b > a) out[batch * out_h * out_w * f_in_ch + tii * out_h * out_w + trr * out_h + tcc] = b;

												}
											}

									}
								}
							}
						}
					}

			}
		}
	}
}
void conv_bias_relu(float* in, float* weight, float* bias, float* out, int f_W,int f_H,int w_k,int f_out_ch,int f_in_ch,int relu_en)
{
	//in[batch*f_in_ch*f_W*f_H]

	//weight[w_out_ch*w_in_ch*w_k*w_k]

	//out[batch*f_out_ch*out_w*out_h]

	int out_w = (f_W - w_k) + 1;
	int out_h = (f_H - w_k) + 1;
	for (int batch = 0; batch < BATCH; batch++)
	{
		for (int to = 0; to < f_out_ch; to++)
		{
			for (int tw = 0; tw < out_w; tw++)
			{
				for (int th = 0; th < out_h; th++)
				{
					float sum = 0;
					for (int ii = 0; ii < w_k; ii++)
					{
						for (int jj = 0; jj < w_k; jj++)
						{
							int h = th + jj;
							int w = tw  + ii;
							if ((h >= 0 )&&( w >= 0 )&&( h < f_H) &&( w < f_W))
							{
								for (int ti = 0; ti < f_in_ch; ti++)
								{
									float tp;
									tp = in[batch * f_in_ch * f_W * f_H + ti * f_W * f_H + w * f_H + h] *
										weight[to * f_in_ch * w_k * w_k + ti * w_k * w_k + ii * w_k + jj];
									//if(relu_en==2)std:: cout<< tp << " "<<std::endl;
									sum += tp;
								}
							}
						}
					}


					sum += bias[to];
					if (relu_en & (sum < 0))
						sum = 0;

					out[batch*f_out_ch*out_w*out_h+to * out_w * out_h+tw * out_h + th] = sum;

				}
			}
		}
	}
}
void batchnorm_no(float *in, float *out, float *gamma, float *beta, int w,int h,int cout)
{
	for (int tw = 0;tw < w;tw++)
	{
		for (int th = 0;th < h;th++)
		{
			for (int ti = 0;ti < cout;ti++)
			{
				out[ti * w * h + tw * h + th] = gamma[ti] * in[ti * w * h + tw * h + th] + beta[ti];
				if(out[ti * w * h + tw * h + th]<0) out[ti * w * h + tw * h + th] =0;
			}
		}
	}

}

int main()
{
	float *in1=new float[1219164];
	float *feature=new float[3*320*160];
	float *feature2=new float[3*320*160];
	float *pad=new float[3*322*162];
	load_data("F:/contest/dss/feature.bin", (char*)feature, 3*320*160*4);
	float *weight=new float[211632];
	load_data("F:/contest/dss/w.bin", (char*)weight, 211632*4);
	float *gamma=new float[432];
	load_data("F:/contest/dss/newga.bin", (char*)gamma, 432*4);
	float *beta=new float[432];
	load_data("F:/contest/dss/newbe.bin", (char*)beta, 432*4);

	for(int k=0;k<3;k++)
	{
		for(int j=0;j<160;j++)
		{
			for(int i=0;i<320;i++)
			{
				feature2[k*320*160+i*160+j]=feature[k*160*320+j*320+i];
			}
		}
	}
	std::cout<<feature2[0]<<std::endl;
	padding_no(feature2,pad,320,160,3,0);

	for(int i=0;i<3*322*162;i++)
	{
		in1[i]=pad[i];
	}
	for(int i=0;i<211632;i++)
	{
		in1[i+3*322*162]=weight[i];

	}
	for(int i=0;i<432;i++)
	{
		in1[i+3*322*162+211632]=gamma[i];
		in1[i+3*322*162+211632+432]=beta[i];

	}

	float temp_feature1[CONV_Tin][CONV_Tw][CONV_Th];
	float temp_feature2[CONV_Tin][CONV_Tw-2][CONV_Th-2];
	float temp_weight1[CONV_Tout][CONV_Tin][K][K];
	get_block_weight3x3(weight, temp_weight1, 0, 0,16,3,3);
	get_block_feature3x3(pad, temp_feature1,0, 0,0,322,162,3);
	basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
	float bias[16]={0};
	float *out=new float[16*320*160];


	top(in1);

	float *ans=new float[12000];
	std::cout<<std::endl;
	for(int k=0;k<60;k++)
	{
		for(int i=0;i<20;i++)
		{
			for(int j=0;j<10;j++)
			{
				ans[k*20*10+i*10+j]=in1[156492+211632+432+432+425088+k*20*10+i*10+j];
				//if(out3[k*160*80+i*80+j]!=out4[k*160*80+i*80+j])std::cout<<k<<" "<<i<<" "<<j<<" different"<<std::endl;
			}
		}
	}
	for(int k=0;k<1;k++)
	{
		for(int j=0;j<10;j++)
		{
			for(int i=0;i<20;i++)
			{
				std::cout<<ans[k*20*10+i*10+j]<<" ";
				//if(out3[k*160*80+i*80+j]!=out4[k*160*80+i*80+j])std::cout<<k<<" "<<i<<" "<<j<<" different"<<std::endl;
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}
	std::cout<<"over"<<std::endl;
	return 0;
}
