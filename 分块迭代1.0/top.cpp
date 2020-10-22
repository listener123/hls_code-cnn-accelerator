#include "top.h"
#include <iostream>

//offset is used to move to next layer
template<unsigned w_in,unsigned w_out,unsigned w_k>
void load_conv_weight(float *in, float* out,int offset)
{
	for (int i = 0; i < w_k;i++)
	{
		for (int j = 0; j < w_k;j++)
		{
			load_conv_weight_label1:for (int ti = 0; ti < w_in;ti++)
			{
				load_conv_weight_label0:for (int to = 0; to < w_out;to++)
				{
					out[to * w_in * w_k * w_k + ti * w_k * w_k + i * w_k + j] = in[offset+i * w_out * w_in * w_k + j * w_out * w_in + ti * w_out + to];
				}
			}
		}
	}
}
template<unsigned w_in>
void load_conv_bias(float *in, float* out,int offset)
{

	load_conv_bias_label2:for (int ti = 0; ti < w_in;ti++)
	{
		out[ti]=in[offset+ti];
	}

}
template<unsigned f_W,unsigned f_H,unsigned f_in_ch,unsigned value>
void padding(float *in, float *out)
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
#pragma HLS PIPELINE
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
template<unsigned a>
void clean(float *in)
{
	clean_label3:for (int i = 0; i < a; i++)
	{
		in[i] = 0;
	}

}

//conv3x3
template<unsigned W,unsigned H,unsigned IN>
void get_block_feature3x3(float *in, float *out,int cin,int row,int col)
{
	int start_in = cin;
	int start_w = row;
	int start_h = col;

	get_block_feature_label2:for (int tw = 0; tw < CONV_Tw; tw++)
	{
		get_block_feature_label1:for (int th = 0; th < CONV_Th; th++)
		{
			get_block_feature_label0:for (int ti = 0; ti < CONV_Tin; ti++)
			{

				if(tw+start_w<W&&th+start_h<H&&ti+start_in<IN)
				out[ti*CONV_Tw*CONV_Th+tw*CONV_Th+th] =in[(ti+start_in)*W*H+(tw+start_w)*H+(th+start_h)];
				else out[ti*CONV_Tw*CONV_Th+tw*CONV_Th+th] = 0;
			}
		}
	}
}
template<unsigned OUT,unsigned IN,unsigned k>
void get_block_weight3x3(float *in, float *out, int cout, int cin)
{
	int start_in = cin;
	int start_out = cout;

	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < K; j++)
		{
			get_block_weight_label9:for (int ti = 0; ti < CONV_Tin; ti++)
			{
				get_block_weight_label8:for (int to = 0; to < CONV_Tout; to++)
				{
					if(to+start_out<OUT&&ti+start_in<IN)
					out[(to)*CONV_Tin*K*K+(ti)*K*K+i*K+j] = in[(to+start_out)*IN*k*k+(ti+start_in)*k*k+i*k+j];
					else out[(to)*CONV_Tin*K*K+(ti)*K*K+i*K+j] = 0;
				}
			}
		}
	}
}
template<unsigned W,unsigned H,unsigned OUT>
void back_result3x3(float *in, float *out, int cout, int row, int col)
{
	int start_out = cout;
	int start_w = row;
	int start_h = col;
	for (int tw = 0; tw <CONV_Tw; tw++)
	{
		back_result_label11:for (int th = 0; th < CONV_Th; th++)
		{
			back_result_label10:for (int to = 0; to < CONV_Tout; to++)
			{
				if (tw+start_w < W && th+start_h < H && to+start_out < OUT && (tw<CONV_Tw-1) && (th<CONV_Th-1))
				{
					out[(to+start_out)*W*H+(tw+start_w)*H+th+start_h] = in[to*CONV_Tw*CONV_Th+(tw+1)*CONV_Th+th+1];
				}
			}
		}
	}
}

void basic_conv3x3(float in[CONV_Tin*CONV_Tw*CONV_Th], float weight[CONV_Tout*CONV_Tin*K*K], float out[CONV_Tout*CONV_Tw*CONV_Th])
{


	for (int tw = 1; tw < CONV_Tw-1; tw++)
	{
		for (int th = 1; th < CONV_Th-1; th++)
		{
			basic_conv3x3_label12:for (int to = 0; to < CONV_Tout; to++)
			{
#pragma HLS UNROLL
				float tp=0;
				for (int ii = 0; ii < K; ii++)
				{
					for (int jj = 0; jj < K; jj++)
					{
						int h = th  + jj - 1;
						int w = tw  + ii - 1;
						basic_conv3x3_label13:for (int ti = 0; ti < CONV_Tin; ti++)
						{
#pragma HLS PIPELINE
							tp += in[ti*CONV_Tw*CONV_Th+w*CONV_Th+h] *weight[to*CONV_Tin*K*K+ti*K*K+ii*K+jj];
						}
					}
				}
				out[to*CONV_Tw*CONV_Th+tw*CONV_Th+th]=tp;
			}
		}
	}
}
template<unsigned cin>
void add_block3x3(float in[cin][CONV_Tin*CONV_Tw*CONV_Th],float *out,int cint)
{	for(int j=0;j<CONV_Tin*CONV_Tw*CONV_Th;j++)
	{
		float tp=0;
		for(int i=0;i<cint;i++)
		{
			tp+=in[i][j];
		}
		out[j]=tp;
	}
}
template<unsigned w,unsigned h,unsigned cout,unsigned cin>
void conv3x3_block(float *in,float *weight,float *out)
{
	float temp_weight1[CONV_Tout*CONV_Tin*K*K];
	float temp_feature1[CONV_Tin*CONV_Tw*CONV_Th];
	float temp_feature3[CONV_Tin*CONV_Tw*CONV_Th];
	//flag=0;
	//clean<CONV_Tin*CONV_Tw*CONV_Th>(temp_feature2);
	int row_num = 0;
	int col_num = 0;
	int cin_num = 0;
	int cout_num = 0;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;
	float temp_feature2[cin / CONV_Tin+1][CONV_Tin*CONV_Tw*CONV_Th];

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			conv3x3_block_label0:for (int cou = 0; cou < cout_num; cou++)
			{
#pragma HLS DATAFLOW
				conv3x3_block_label14:for (int ci = 0; ci < cin_num; ci++)
				{
#pragma HLS DATAFLOW
					get_block_weight3x3<cout,cin,3>(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin);
					get_block_feature3x3<w,h,cin>(in, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2) );
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2[ci]);
				}
				add_block3x3<cin / CONV_Tin+1>(temp_feature2,temp_feature3,cin_num);
				back_result3x3<w-2,h-2,cout>(temp_feature3, out, cou * CONV_Tout, ro * (CONV_Tw - 2), co * (CONV_Th - 2));
				//clean_data<CONV_Tin*CONV_Tw*CONV_Th>(temp_feature2);
			}
		}
	}
}
//pool
template<int IN,int W,int H>
void get_block_feature_POOL(float *in, float *out, int col, int row,int cin)
{
	int start_in = cin;
	int start_w = row;
	int start_h = col;

	for (int tw = 0; tw < POOL_Tw; tw++)
	{
		for (int th = 0; th < POOL_Th; th++)
		{
			for (int ti = 0; ti < POOL_Tout; ti++)
			{

					if(tw+start_w<W&&th+start_h<H&&ti+start_in<IN)
						out[(ti) * POOL_Tw * POOL_Th + (tw) * POOL_Th + (th)] =
							in[(ti + start_in) * W * H + (tw+start_w) * H + th+start_h];
					else
						out[(ti) * POOL_Tw * POOL_Th + (tw) * POOL_Th + (th)]=-99999;

			}
		}
	}
}
template<unsigned OUT>
void get_block_bias_POOL(float *in, float *out, int cout)
{
	int start = cout;
	for (int i = 0; i < POOL_Tout; i++)
	{
		if(i+start<OUT)
			out[i] = in[i+start];
		else out[i]=0;
	}
}
template<unsigned OUT,unsigned W,unsigned H>
void back_result_POOL(float *in, float *out, int cout, int row, int col)
{
	//in[POOL_Tout * (POOL_Tw/2) * (POOL_Th/2)]
	//out[OUT*W*H]
	int start_out = cout;
	int start_w = row;
	int start_h = col;
	for (int tw = 0; tw < (POOL_Tw/2); tw++)
	{
		for (int th = 0; th < (POOL_Th/2); th++)
		{
			for (int to = 0; to < POOL_Tout; to++)
			{
					if(tw+start_w<W&&th+start_h<H&&to+start_out<OUT)
					out[(to+start_out) * W * H + (tw+start_w) * H + th+start_h] =
						in[to * (POOL_Tw/2) * (POOL_Th/2) + tw * (POOL_Th/2) + th];
			}
		}
	}
}
void basic_bias_relu_maxpool(float in[POOL_Tout*POOL_Tw*POOL_Th],  float bias[POOL_Tout], float out[POOL_Tout*(POOL_Tw/2)*(POOL_Th/2)])
{
	//bias and  relu
	float temp_out[BATCH * POOL_Tout * POOL_Tw * POOL_Th] = { 0 };

	for (int to = 0; to < POOL_Tout; to++)
	{

		for (int t = 0; t < POOL_Tw*POOL_Th; t++)
		{

				temp_out[to * POOL_Tw * POOL_Th + t] =in[to * POOL_Tw * POOL_Th + t] + bias[to];
				if (temp_out[to * POOL_Tw * POOL_Th + t] < 0)  temp_out[to * POOL_Tw * POOL_Th + t] = 0;
		}
	}


	for (int tw = 0; tw < POOL_Tw/2; tw++)
	{
		for (int th = 0; th < POOL_Th/2; th++)
		{

			for (int to = 0; to < POOL_Tout; to++)
			{
				float element1 = temp_out[to * POOL_Tw * POOL_Th + 2*tw * POOL_Th + 2*th];
				float element2 = temp_out[to * POOL_Tw * POOL_Th + (2*tw+1) * POOL_Th + 2*th];
				float element3 = temp_out[to * POOL_Tw * POOL_Th + (2*tw) * POOL_Th + 2*th+1];
				float element4 = temp_out[to * POOL_Tw * POOL_Th + (2*tw + 1) * POOL_Th + 2*th + 1];
				float result1 = (element1 > element2) ? element1 : element2;
				float result2 = (element3 > element4) ? element3 : element4;
				out[to *(POOL_Tw/2)*(POOL_Th/2) + tw * (POOL_Th/2) + th] = (result1 > result2) ? result1 : result2;
			}
		}
	}
}
template<unsigned w,unsigned h,unsigned cout>
void bias_relu_pool(float *in,float *bias,float *out)
{
	float temp_feature1[POOL_Tout*POOL_Tw*POOL_Th];
	float temp_feature2[POOL_Tout*POOL_Tw*POOL_Th];
	float temp_bias1[POOL_Tout];
	//处理4*8*8的特征图，分块2*4*4，进行pool
	int row_num = 0;
	if(w%POOL_Tw!=0)row_num=w / POOL_Tw+1;
	else row_num=w / POOL_Tw;
	int col_num = 0;
	if(h%POOL_Th!=0)col_num=h / POOL_Th+1;
		else col_num=h / POOL_Th;
	int cin_num = 0;
	if(cout%POOL_Tout!=0)cin_num=cout / POOL_Tout+1;
		else cin_num=cout / POOL_Tout;

	for (int ro = 0;ro < row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			bias_relu_pool_label1:for (int ci = 0; ci < cin_num; ci++)
			{
#pragma HLS DATAFLOW
				get_block_feature_POOL<cout,w,h>(in, temp_feature1, co * POOL_Th, ro * POOL_Tw, ci * POOL_Tout);
				get_block_bias_POOL<cout>(bias, temp_bias1, ci * POOL_Tout);
				basic_bias_relu_maxpool(temp_feature1,temp_bias1, temp_feature2);
				//输出到结果
				back_result_POOL<cout,w/2,h/2>(temp_feature2, out, ci * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2));

			}
		}
	}
}
//conv1X1
template<unsigned IN,unsigned W,unsigned H>
void get_block_feature_1x1(float *in, float *out, int col, int row,int cin)
{
	int start_in = cin;
	int start_w = row;
	int start_h = col;

	for (int tw = 0; tw < CONV1x1_Tw; tw++)
	{
		for (int th = 0; th < CONV1x1_Th; th++)
		{
			for (int ti = 0; ti < CONV1x1_Tin; ti++)
			{

				//if (batch * IN * W * H + ti * W * H + tw * H + th < BATCH * IN * W * H)
				if(tw+start_w<W&&th+start_h<H&&ti+start_in<IN)
				out[ti * CONV1x1_Tw * CONV1x1_Th + tw * CONV1x1_Th + th] =
						in[(ti+start_in) * W * H + (tw+start_w) * H + th+start_h];
				else out[ti * CONV1x1_Tw * CONV1x1_Th + tw * CONV1x1_Th + th] = 0;

			}
		}
	}
}
template<unsigned OUT, unsigned IN >
void get_block_weight_1x1(float *in, float *out, int cout, int cin)
{
	int start_in = cin;
	int start_out = cout;

	for (int ti = 0; ti < CONV1x1_Tin; ti++)
	{
		for (int to = 0; to < CONV1x1_Tout; to++)
		{
//			printf("ti: %d, to: %d, pos: %d, val: %.0f\n", ti, to, to * IN + ti, in[to * IN + ti]);
			if(to+start_out<OUT&&ti+start_in<IN)
			out[to * CONV1x1_Tin + ti] = in[(to+start_out) * IN + ti+start_in];
			else out[to * CONV1x1_Tin + ti] = 0;
		}
	}
}
void basic_conv_1x1(float in[CONV1x1_Tin*CONV1x1_Tw*CONV1x1_Th], float weight[CONV1x1_Tout*CONV1x1_Tin*1*1], float out[CONV1x1_Tout*CONV1x1_Tw*CONV1x1_Th])
{

		for (int tw = 0; tw < CONV1x1_Tw; tw++)
		{
			for (int th = 0; th < CONV1x1_Th; th++)
			{
				for (int to = 0; to < CONV1x1_Tout; to++)
				{
					float tp=0;
					for (int ti = 0; ti < CONV1x1_Tin; ti++)
					{
						tp += in[ti * CONV1x1_Tw * CONV1x1_Th + tw * CONV1x1_Th + th] * weight[to * CONV1x1_Tin + ti];
					}
					out[ to * CONV1x1_Tw * CONV1x1_Th + tw * CONV1x1_Th + th]=tp;
				}

			}
		}

}
template<unsigned OUT,unsigned W,unsigned H>
void back_result_1x1(float *in, float *out, int cout, int row, int col)
{
	int start_out = cout;
	int start_w = row;
	int start_h = col;
	for (int tw = 0; tw < 0 + POOL_Tw; tw++)
	{
		for (int th = 0; th < 0 + POOL_Th; th++)
		{
			for (int to = 0; to < 0 + POOL_Tout; to++)
			{

					if(tw +start_w< W && th+start_h < H && to+start_out < OUT)
					out[(to+start_out) * W * H + (tw+start_w) * H + th+start_h] =
						in[to * CONV1x1_Tw * CONV1x1_Th + tw * CONV1x1_Th + th];

			}
		}
	}
}

template<unsigned cin>
void add_block1x1(float in[cin][CONV1x1_Tin*CONV1x1_Tw*CONV1x1_Th],float *out,int cint)
{	for(int j=0;j<CONV1x1_Tin*CONV1x1_Tw*CONV1x1_Th;j++)
	{
		float tp=0;
		for(int i=0;i<cint;i++)
		{
			tp+=in[i][j];
		}
		out[j]=tp;
	}
}

template<unsigned w,unsigned h,unsigned cin,unsigned cout>
void conv_1x1(float *in,float *weight,float *out)
{
	float temp_weight1[CONV1x1_Tout*CONV1x1_Tin*K*K];
	float temp_feature1[CONV1x1_Tout*CONV1x1_Tw*CONV1x1_Th];
	float temp_feature3[CONV1x1_Tout*CONV1x1_Tw*CONV1x1_Th];
	float temp_feature2[cin / CONV1x1_Tin + 1][CONV1x1_Tout*CONV1x1_Tw*CONV1x1_Th];
	int row_num = 0;
	int col_num = 0;
	int cin_num = 0;
	int cout_num = 0;
	if(w % CONV1x1_Tw !=0) row_num = w / CONV1x1_Tw + 1;
		else row_num = w / CONV1x1_Tw;
	if(h % CONV1x1_Th !=0) col_num = h / CONV1x1_Th + 1;
		else col_num = h / CONV1x1_Th;
	if(cin % CONV1x1_Tin != 0) cin_num = cin / CONV1x1_Tin + 1;
		else cin_num = cin / CONV1x1_Tin;
	if(cout % CONV1x1_Tout != 0) cout_num = cout / CONV1x1_Tout + 1;
		else cout_num = cout / CONV1x1_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{

				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight_1x1<cout,cin>(weight, temp_weight1, cou*CONV1x1_Tout, ci*CONV1x1_Tin);
					get_block_feature_1x1<cin,w,h>(in, temp_feature1,  co*CONV1x1_Th, ro*CONV1x1_Tw, ci * CONV1x1_Tin);
					basic_conv_1x1(temp_feature1, temp_weight1, temp_feature2[ci]);
				}
				add_block1x1<cin / CONV_Tin+1>(temp_feature2,temp_feature3,cin_num);
				back_result_1x1<cout,w,h>(temp_feature3, out, cou * CONV1x1_Tout, ro * CONV1x1_Tw, co * CONV1x1_Th);
			}
		}
	}
}
//BN
void basic_batchnorm(float in[BN_Tin * BN_Tw * BN_Th], float out[BN_Tin * BN_Tw * BN_Th], float gamma[BN_Tin], float beta[BN_Tin], float mean[BN_Tin], float var[BN_Tin])
{

	for (int ti = 0;ti < BN_Tin;ti++)
	{
		for (int tw = 0;tw < BN_Tw;tw++)
		{
			for (int th = 0;th < BN_Th;th++)
			{
				out[ti * BN_Tw * BN_Th + tw * BN_Th + th] = gamma[ti] * (in[ti * BN_Tw * BN_Th + tw * BN_Th + th] - mean[ti]) / (var[ti] + eps) + beta[ti];
			}
		}
	}

}
template<unsigned IN,unsigned W,unsigned H>
void get_block_feature_BN(float *in,float *out,int col,int row,int cin)
{
	for (int tw = 0;tw < 0 + BN_Tw;tw++)
	{
		for (int th = 0;th < 0 + BN_Th;th++)
		{
			for (int ti = 0;ti < 0 + BN_Tin;ti++)
			{
				if (tw+row < W && th+col < H && ti+cin < IN)
				{
					out[ti * BN_Tw * BN_Th + tw * BN_Th + th] = in[(ti+cin) * W * H + (tw+row) * H + th+col];
				}
				else out[ti * BN_Tw * BN_Th + tw * BN_Th + th] =0;
			}
		}
	}
}
template<unsigned IN>
void get_block_weight_BN(float* in_g, float* in_b, float* in_m, float* in_v, float gamma[BN_Tin], float beta[BN_Tin], float mean[BN_Tin], float var[BN_Tin],int cin)
{
	for (int i = 0;i < BN_Tin;i++)
	{
		if (i+cin < IN)
		{
			gamma[i] = in_g[i+cin];
			beta [i] = in_b[i+cin];
			mean [i] = in_m[i+cin];
			var  [i] = in_v[i+cin];
		}
		else
		{
			gamma[i] = 0;
			beta [i] = 0;
			mean [i] = 0;
			var  [i] = 0;
		}
	}
}
template<unsigned OUT, unsigned W, unsigned H>
void back_result_BN(float *in, float* out, int cout, int row, int col)
{
	for (int tw = 0;tw < 0 + BN_Tw;tw++)
	{
		for (int th = 0;th < 0 + BN_Th;th++)
		{
			for (int to = 0;to < 0 + BN_Tin;to++)
			{
				if (tw+row < W && th+col < H && to+cout < OUT)
					out[(to+cout) * W * H + (tw+row) * H + th+col] = in[(to) * BN_Tw * BN_Th + (tw) * BN_Th + (th)];
			}
		}
	}
}
template<unsigned w,unsigned h,unsigned cout>
void batchnorm(float* in, float* out, float* gamma, float* beta, float* mean, float* var)
{
	float temp_weight1[BN_Tin*CONV1x1_Tin*K*K];
	float temp_feature1[BN_Tin*BN_Tw*BN_Th];
	float temp_feature2[BN_Tin*BN_Tw*BN_Th];
	float temp_gamma[BN_Tin];
	float temp_beta[BN_Tin];
	float temp_mean[BN_Tin];
	float temp_var[BN_Tin];
	int row_num = 0;
	if (w % BN_Tw != 0)row_num = w / BN_Tw + 1; else row_num = w / BN_Tw;
	int col_num = 0;
	if (h % BN_Th != 0)col_num = h / BN_Th + 1; else col_num = h / BN_Th;
	int cin_num = 0;
	if (cout % BN_Tin != 0)cin_num = cout / BN_Tin + 1; else cin_num = cout / BN_Tin;

	for (int ro = 0;ro < row_num;ro++)
	{
		for (int co = 0;co < col_num;co++)
		{
			batchnorm_label0:for (int ci = 0;ci < cin_num;ci++)
			{
#pragma HLS DATAFLOW
				get_block_feature_BN<cout,w,h>(in,temp_feature1,co*BN_Th,ro*BN_Tw,ci*BN_Tin);
				get_block_weight_BN<cout>(gamma,beta,mean,var,temp_gamma,temp_beta,temp_mean,temp_var,ci*BN_Tin);
				basic_batchnorm(temp_feature1, temp_feature2, temp_gamma, temp_beta, temp_mean, temp_var);

				back_result_BN<cout,w,h>(temp_feature2,out, ci * BN_Tin, ro * BN_Tw, co * BN_Th);
			}
		}
	}
}


template<unsigned f_W,unsigned f_H,unsigned w_k,unsigned f_out_ch,unsigned f_in_ch>
void conv(float *in, float *weight,float *bias, float *out)
{
	//in[batch*f_in_ch*f_W*f_H]

	//weight[w_out_ch*w_in_ch*w_k*w_k]

	//out[batch*f_out_ch*out_w*out_h]

	int out_w = (f_W - w_k) + 1;
	int out_h = (f_H - w_k) + 1;
	for (int to = 0; to < f_out_ch; to++)
	{
		for (int tw = 0; tw < out_w; tw++)
		{
			conv_label8:for (int th = 0; th < out_h; th++)
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
							conv_label3:for (int ti = 0; ti < f_in_ch; ti++)
							{
#pragma HLS UNROLL
								float tp;
								tp = in[ti*f_W*f_H+w*f_H+h] * weight[to*f_in_ch*w_k*w_k+ti*w_k*w_k+ii*w_k+jj];
								//if(relu_en==2)std:: cout<< tp << " "<<std::endl;
								sum += tp;
							}
						}
					}
				}

				sum += bias[to];
				if ((sum < 0))
					sum = 0;
				out[to*out_w*out_h+tw*out_h+th] = sum;
			}
		}
	}
}
template<unsigned f_W,unsigned f_H,unsigned w_k,unsigned f_in_ch,unsigned type>
void pool(float *in, float *out)
{
	//in[f_in_ch*f_W*f_H]

	//out[f_in_ch*out_w*out_h]
	//type0 max type1 average
	//compute
	int out_w = (f_W - w_k) / 2 + 1;
	int out_h = (f_H - w_k) / 2 + 1;
	for (int row = 0; row < out_w; row++)
	{
		pool_label7:for (int col = 0; col < out_h; col++)
		{
			pool_label6:for (int ti = 0; ti < f_in_ch; ti++)
			{
				pool_label5:for (int i = 0; i < w_k; i++)
				{
					pool_label4:for (int j = 0; j < w_k; j++)
					{

						if (type == 1) //average pool
						{
							out[ti*out_w*out_h+row*out_h+col] +=in[ti*f_W*f_H+(2 * row + i)*f_H+2 * col + j] / (w_k * w_k);
						}
						if (type == 0) //max pool
						{
							float a = out[ti*out_w*out_h+row*out_h+col];
							float b = in[ti*f_W*f_H+(2 * row + i)*f_H+2 * col + j];
							if (b > a) out[ti*out_w*out_h+row*out_h+col] = b;
						}
					}
				}
			}
		}
	}
}

//weight 16*1*3*3+32*16*3*3+128*32*7*7+10*128*1*1
//bias 16+32+128+10
//temp 16*28*28
void top(float in[1*28*28],float weight[206736],float bias[186],float out[10],float temp1[12544],float temp2[12544])
{
#pragma HLS INTERFACE m_axi depth=4294967295 port=temp1 offset=slave bundle=TEMP1
#pragma HLS INTERFACE m_axi depth=4294967295 port=temp2 offset=slave bundle=TEMP2
#pragma HLS INTERFACE m_axi depth=10*1 port=out offset=slave bundle=DATA_OUT
#pragma HLS INTERFACE m_axi depth=4294967295 port=bias offset=slave bundle=BIAS
#pragma HLS INTERFACE m_axi depth=4294967295 port=weight offset=slave bundle=WEIGHT
#pragma HLS INTERFACE m_axi depth=28*28*1 port=in offset=slave bundle=DATA_IN
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL

	float weight_[200704];//128*32*7*7 max memory in lenet
	float bias_[128];//128 max memory in lenet
	int offset_weight=0;
	int offset_bias=0;
	//initialize
	clean<128*32*7*7>(weight_);
	clean<128>(bias_);
	//bias_relu_pool<28,28,16>(temp2,bias_,temp1);

//conv1
	std::cout<<"start"<<std::endl;
	padding<28,28,1,0>(in,temp1);
	load_conv_weight<1,16,3>(weight, weight_, offset_weight);
	load_conv_bias<16>(bias,bias_,offset_bias);
	//conv<30,30,3,16,1>(temp1,weight_,bias_,temp2);
	conv3x3_block<30,30,16,1>(temp1,weight_,temp2);
	std::cout<<"conv1"<<std::endl;
	//clean<12544>(temp1);
	//pool<28,28,2,16,0>(temp2,temp1);
	bias_relu_pool<28,28,16>(temp2,bias_,temp1);
	//clean<12544>(temp2);
	//clean<128*32*7*7>(weight_);
	//clean<128>(bias_);
	std::cout<<"pool1"<<std::endl;
//conv2
	padding<14,14,16,0>(temp1,temp2);
	offset_weight+=1*16*3*3;
	offset_bias+=16;
	load_conv_weight<16,32,3>(weight, weight_, offset_weight);
	load_conv_bias<32>(bias,bias_,offset_bias);
	//conv<16,16,3,32,16>(temp2,weight_,bias_,temp1);
	conv3x3_block<16,16,32,16>(temp2,weight_,temp1);
	clean<12544>(temp2);
	//pool<14,14,2,32,0>(temp1,temp2);
	bias_relu_pool<14,14,32>(temp1,bias_,temp2);
	//clean<12544>(temp1);
	//clean<128*32*7*7>(weight_);
	//clean<128>(bias_);

//fc1
	offset_weight+=16*32*3*3;
	offset_bias+=32;
	load_conv_weight<32,128,7>(weight, weight_, offset_weight);
	load_conv_bias<128>(bias,bias_,offset_bias);
	conv<7,7,7,128,32>(temp2,weight_,bias_,temp1);
	//clean<12544>(temp2);
	//clean<128*32*7*7>(weight_);
	//clean<128>(bias_);
//fc2
	offset_weight+=32*128*7*7;
	offset_bias+=128;
	load_conv_weight<128,10,1>(weight, weight_, offset_weight);
	load_conv_bias<10>(bias,bias_,offset_bias);
	conv<1,1,1,10,128>(temp1,weight_,bias_,out);
	//clean<12544>(temp1);
	//clean<128*32*7*7>(weight_);
	//clean<128>(bias_);

}
