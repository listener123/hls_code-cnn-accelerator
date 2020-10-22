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
int main()
{/*
	float in[1 * 28 * 28];
	load_data("F:/test_block/data/test_data.bin", (char*)in, sizeof(in));

	float weight[144];
	float bias[16];
	for(int i=0;i<16;i++)
	{
		bias[i]=i;
		for(int j=0;j<3;j++)
		{
			for(int k=0;k<3;k++)
			{
				weight[i*3*3+j*3+k]=i*3*3+j*3+k;
			}
		}
	}
	float in1[16*28*28];
	float in2[16*28*28];
	for(int i=0;i<28;i++)
	{
		for(int j=0;j<28;j++)
			{
			for(int k=0;k<16;k++)
				{
					in1[k*28*28+i*28+j]=k*28*28+i*28+j+bias[k];
					in2[k*28*28+i*28+j]=k*28*28+i*28+j;

				}
			}

	}
	float *out =new float [16*28*28];
	for(int i=0;i<144;i++) out[i]=0;
	float *out2=new float [16*28*28];
	float feature_1[16*28*28]={0};
	float feature_2[16*28*28];
	float pad[1*30*30];
	//pool_no(in1,out,28,28,2,16,0);
	//bias_relu_pool<28,28,1>(in,bias,out2);
	//top(in2,weight,bias,out,feature_1,feature_2);

	padding<28,28,1,0>(in,pad);
	conv_bias_relu(pad,weight,bias,out,30,30,3,16,1,1);
	conv3x3_block<30,30,16,1>(pad,weight,out2);
	for(int k=0;k<16;k++)
	{
		for(int i=0;i<28;i++)
		{
			for(int j=0;j<28;j++)
			{
				if(out[k*28*28+i*28+j]!=out2[k*28*28+i*28+j]+bias[k]) std::cout<<k<<" "<<i<<" "<<j<<" different"<<std::endl;
			}
		}
	}*/


	//initialize feature
	float in[1 * 28 * 28];
	load_data("F:/test_block/data/test_data.bin", (char*)in, sizeof(in));

	for(int i=0;i<1;i++)
	{
		for(int j=0;j<28;j++)
		{
			for(int k=0;k<28;k++)
			{
				in[i*28*28+j*28+k]=in[0*28*28+j*28+k];
			}
		}
	}

	//initialize weight and bias
	float *weight_all=new float[1*16*3*3+16*32*3*3+32*128*7*7+128*10*1*1];
	float *bias_all=new float[16+32+128+10];

	float w1[3 * 3 * 1 * 16];
	load_data("F:/test_block/data/W_conv1.bin", (char*)w1, sizeof(w1));
	float w2[3 * 3 * 16 * 32];
	load_data("F:/test_block/data/W_conv2.bin", (char*)w2, sizeof(w2));
	float *w3=new float[7 * 7 * 32 * 128];
	load_data("F:/test_block/data/W_fc1.bin", (char*)w3, 7 * 7 * 32 * 128 * 4);
	float w4[1 * 1 * 128 * 10];
	load_data("F:/test_block/data/W_fc2.bin", (char*)w4, sizeof(w4));

	for(int i=0;i<3*3*1*16;i++) weight_all[i]=w1[i];
	for(int i=0;i<3*3*16*32;i++) weight_all[3*3*1*16+i]=w2[i];
	for(int i=0;i<7*7*32*128;i++) weight_all[3*3*1*16+3*3*16*32+i]=w3[i];
	for(int i=0;i<1*1*128*10;i++) weight_all[3*3*1*16+3*3*16*32+7*7*32*128+i]=w4[i];

	float b1[16];
	load_data("F:/test_block/data/b_conv1.bin", (char*)b1, sizeof(b1));
	float b2[32];
	load_data("F:/test_block/data/b_conv2.bin", (char*)b2, sizeof(b2));
	float b3[128];
	load_data("F:/test_block/data/b_fc1.bin", (char*)b3, sizeof(b3));
	float b4[10];
	load_data("F:/test_block/data/b_fc2.bin", (char*)b4, sizeof(b4));

	for(int i=0;i<16;i++) bias_all[i]=b1[i];
	for(int i=0;i<32;i++) bias_all[16+i]=b2[i];
	for(int i=0;i<128;i++) bias_all[16+32+i]=b3[i];
	for(int i=0;i<10;i++) bias_all[16+32+128+i]=b4[i];

	//result
	float out[10]={0};
	float feature_1[16*28*28];
	float feature_2[16*28*28];
	top(in,weight_all,bias_all,out,feature_1,feature_2);
	for (int i=0;i<1;i++)
	{
		for(int j=0;j<10;j++)
			std::cout<<out[i*10+j]<<" ";
		std::cout<<std::endl;
	}
	std::cout<<std::endl;

	return 0;
}
