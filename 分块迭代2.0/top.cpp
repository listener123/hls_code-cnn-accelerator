#include "top.h"
#include <iostream>
#include <string.h>
int offset_weight;
int offset_bias;
template<unsigned cin,unsigned w,unsigned h>
void clean(float a[cin][w][h])
{
	for(int i=0;i<cin;i++)
	{
		for(int j=0;j<w;j++)
		{
#pragma HLS PIPELINE
			for(int k=0;k<h;k++)
			{
#pragma HLS UNROLL
				a[i][j][k]=0;
			}
		}
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

	for (int i = 0; i < out_w; i++)
	{
		for (int j=0;j<p;j++)
		{
			for (int ch = 0; ch < f_in_ch; ch++)
			{
				out[ch * out_w * out_h+i* out_h+j] = value;
			}
		}

		for (int j = p; j < out_h - p; j++)
		{
			for (int ch = 0; ch < f_in_ch; ch++)
			{
				if (i < p || i >=out_w - p) out[ch * out_w * out_h + i * out_h + j] = value;
				else out[ch * out_w * out_h + i * out_h + j] = in[ch * f_W * f_H + (i - p) * f_H + j - p];
			}
		}

		for (int j = out_h-p; j < out_h; j++)
		{
			for (int ch = 0; ch < f_in_ch; ch++)
			{
			out[ch * out_w * out_h + i * out_h + j] = value;
			}

		}
	}

}
template<unsigned cin,unsigned w,unsigned h>
void clean_temp(float in[409600])
{

	for(int j=0;j<w;j+=8)
	{
		for(int i=0;i<cin;i++)
		{
#pragma HLS PIPELINE
			for(int k=0;k<8;k++)
			{
#pragma HLS UNROLL
				if(j+k<w){
				in[i*w*h+(j+k)*h+0]=0;
				in[i*w*h+(j+k)*h+h-1]=0;}
			}
		}
	}

	for(int j=0;j<h;j+=8)
	{
		for(int i=0;i<cin;i++)
		{
#pragma HLS PIPELINE
			for(int k=0;k<8;k++)
			{
#pragma HLS UNROLL
				if(j+k<h){
				in[i*w*h+(0)*h+j+k]=0;
				in[i*w*h+(w-1)*h+j+k]=0;}
			}
		}
	}
}
//conv3x3

void get_block_feature3x3(float *in, float out[CONV_Tin][CONV_Tw][CONV_Th],int cin,int row,int col,unsigned W,unsigned H,unsigned IN)
{
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
	int start_in = cin;
	int start_w = row;
	int start_h = col;

	get_block_feature_label2:for (int tw = 0; tw < CONV_Tw; tw++)
		{
			get_block_feature_label1:for (int th = 0; th < CONV_Th; th++)
			{
	#pragma HLS PIPELINE
				get_block_feature_label0:for (int ti = 0; ti < CONV_Tin; ti++)
				{
					if(tw+start_w<W&&th+start_h<H&&ti+start_in<IN)
					out[ti][tw][th] =in[(ti+start_in)*W*H+(tw+start_w)*H+th+start_h];
					else out[ti][tw][th] = 0;
				}
			}
		}
}
void get_block_weight3x3(float *in, float out[CONV_Tout][CONV_Tin][K][K], int cout, int cin,unsigned OUT,unsigned IN,unsigned k)
{
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
	int start_in = cin;
	int start_out = cout;

	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < K; j++)
		{
			get_block_weight_label9:for (int ti = 0; ti < CONV_Tin; ti++)
			{
#pragma HLS PIPELINE
				get_block_weight_label8:for (int to = 0; to < CONV_Tout; to++)
				{
#pragma HLS UNROLL
//					if(to+start_out<OUT&&ti+start_in<IN)
//					out[to][ti][i][j] = in[offset_weight+(to+start_out)*IN*k*k+(ti+start_in)*k*k+i*k+j];
//					else out[to][ti][i][j] = 0;
					if(to+start_out<OUT&&ti+start_in<IN)
					out[to][ti][j][i] = in[offset_weight+(to+start_out)*IN*k*k+(ti+start_in)*k*k+i*k+j];
					else out[to][ti][j][i] = 0;
				}
			}
		}
	}
}
float mac9(float f1,float f2,float f3,float f4,float f5,float f6,float f7,float f8,float f9,
		   float w1,float w2,float w3,float w4,float w5,float w6,float w7,float w8,float w9
		   )
{
#pragma HLS INLINE off
	float o1=f1*w1;
	float o2=f2*w2;
	float o3=f3*w3;
	float o4=f4*w4;
	float o5=f5*w5;
	float o6=f6*w6;
	float o7=f7*w7;
	float o8=f8*w8;
	float o9=f9*w9;

	float s1=o1+o5;
	float s2=o2+o6;
	float s3=o3+o7;
	float s4=o4+o8;

	float ans1=s1+s3;
	float ans2=s2+s4;

	return ans1+ans2+o9;
}
void basic_conv3x3(float in[CONV_Tin][CONV_Tw][CONV_Th], float weight[CONV_Tout][CONV_Tin][K][K], float out[CONV_Tout][CONV_Tw-2][CONV_Th-2])
{
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
/*
	float window_buf[CONV_Tin][3][3];
#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=1
	float line_buf[CONV_Tin][3][CONV_Th];
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=2

	for (int tw = 0; tw < CONV_Tw; tw++)
	{
		for (int th = 0; th < CONV_Th; th++)
		{
			for (int ti = 0; ti < CONV_Tin; ti++)
			{
#pragma HLS PIPELINE

				float read_in=in[ti][tw][th];
				line_buf[ti][tw % 3][th]=read_in;

				window_buf[ti][2][2] = read_in;
				window_buf[ti][2][1] = line_buf[ti][(tw+2)%3][th];
				window_buf[ti][2][0] = line_buf[ti][(tw+1)%3][th];

				if(tw>=2 && th>=2)
				{
					for (int to = 0; to < CONV_Tout; to++)
					{
#pragma HLS UNROLL
						float tp=out[to][tw-1][th-1];
						tp+=mac9(window_buf[ti][0][0],window_buf[ti][0][1],window_buf[ti][0][2],
								 window_buf[ti][1][0],window_buf[ti][1][1],window_buf[ti][1][2],
								 window_buf[ti][2][0],window_buf[ti][2][1],window_buf[ti][2][2],

								weight[to][ti][0][0],weight[to][ti][0][1],weight[to][ti][0][2],
								weight[to][ti][1][0],weight[to][ti][1][1],weight[to][ti][1][2],
								weight[to][ti][2][0],weight[to][ti][2][1],weight[to][ti][2][2]
								);
						out[to][tw-1][th-1]=tp;
					}
				}
				for(int c=0;c<3;c++)
				{
#pragma HLS UNROLL
					window_buf[ti][0][c]=window_buf[ti][1][c];
					window_buf[ti][1][c]=window_buf[ti][2][c];
				}
			}
		}
	}*/

	for (int tw = 1; tw < CONV_Tw-1; tw++)
	{
		for (int th = 1; th < CONV_Th-1; th++)
		{
			for (int ti = 0; ti < CONV_Tin; ti++)
			{
#pragma HLS PIPELINE II=5
					for (int to = 0; to < CONV_Tout; to++)
					{
#pragma HLS UNROLL
						float tp=out[to][tw-1][th-1];
						tp+=mac9(in[ti][tw-1][th-1],in[ti][tw-1][th],in[ti][tw-1][th+1],
								 in[ti][tw  ][th-1],in[ti][tw  ][th],in[ti][tw  ][th+1],
								 in[ti][tw+1][th-1],in[ti][tw+1][th],in[ti][tw+1][th+1],

								weight[to][ti][0][0],weight[to][ti][0][1],weight[to][ti][0][2],
								weight[to][ti][1][0],weight[to][ti][1][1],weight[to][ti][1][2],
								weight[to][ti][2][0],weight[to][ti][2][1],weight[to][ti][2][2]
//								 weight[to][ti][0][0],weight[to][ti][1][0],weight[to][ti][2][0],
//								 weight[to][ti][0][1],weight[to][ti][1][1],weight[to][ti][2][1],
//								 weight[to][ti][0][2],weight[to][ti][1][2],weight[to][ti][2][2]
								);
						//if(tw-1==2&&th-1==2&&to==0)std::cout<<ti<<" "<<in[ti][tw-1][th-1]<<std::endl;
						out[to][tw-1][th-1]=tp;
					}

			}
		}
	}
	//std::cout<<std::endl;
}

//pool
void back_result_POOL(float in[POOL_Tout][POOL_Tw/2][POOL_Th/2], float *out, int cout, int row, int col,unsigned OUT,unsigned W,unsigned H)
{
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
	//in[POOL_Tout * (POOL_Tw/2) * (POOL_Th/2)]
	//out[OUT*W*H]
	int start_out = cout;
	int start_w = row;
	int start_h = col;
	for (int tw = 0; tw < (POOL_Tw/2); tw++)
	{
		for (int th = 0; th < (POOL_Th/2); th++)
		{
#pragma HLS PIPELINE
			for (int to = 0; to < POOL_Tout; to++)
			{
				if(tw+start_w<W-2&&th+start_h<H-2&&to+start_out<OUT)
					out[(to+start_out) * W * H + (tw+start_w+1) * H + th+start_h+1] =in[to][tw][th];
			}
		}
	}
}
float MAX(float a,float b,float c,float d)
{
#pragma HLS INLINE
	float t1=a>b?a:b;
	float t2=c>d?c:d;
	return t1>t2?t1:t2;
}
void basic_maxpool(float in[POOL_Tout][POOL_Tw][POOL_Th], float out[POOL_Tout][POOL_Tw/2][POOL_Th/2])
{
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
	for (int tw = 0; tw < POOL_Tw/2; tw++)
	{
		for (int th = 0; th < POOL_Th/2; th++)
		{
#pragma HLS PIPELINE II=4
			for (int to = 0; to < POOL_Tout; to++)
			{
#pragma HLS UNROLL
				out[to][tw][th] =MAX(in[to][2*tw  ][2*th],in[to][2*tw  ][2*th+1],
						             in[to][2*tw+1][2*th],in[to][2*tw+1][2*th+1]);
			}
		}
	}
}
//conv1X1
void get_block_feature_1x1(float *in, float out[CONV1x1_Tin][CONV1x1_Tw][CONV1x1_Th], int col, int row,int cin,unsigned IN,unsigned W,unsigned H)
{
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
	int start_in = cin;
	int start_w = row;
	int start_h = col;

	for (int tw = 0; tw < CONV1x1_Tw; tw++)
	{
		for (int th = 0; th < CONV1x1_Th; th++)
		{
#pragma HLS PIPELINE
			for (int ti = 0; ti < CONV1x1_Tin; ti++)
			{

				//if (batch * IN * W * H + ti * W * H + tw * H + th < BATCH * IN * W * H)
				if(tw+start_w<W&&th+start_h<H&&ti+start_in<IN)
				out[ti][tw][th] =
						in[(ti+start_in) * (W+2) * (H+2) + (tw+start_w+1) * (H+2) + th+start_h+1];
				else out[ti][tw][th] = 0;
			}
		}
	}
}
void get_block_weight_1x1(float *in, float out[CONV1x1_Tout][CONV1x1_Tin], int cout, int cin,unsigned OUT, unsigned IN)
{
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
	int start_in = cin;
	int start_out = cout;

	for (int ti = 0; ti < CONV1x1_Tin; ti++)
	{
#pragma HLS PIPELINE
		for (int to = 0; to < CONV1x1_Tout; to++)
		{
//			printf("ti: %d, to: %d, pos: %d, val: %.0f\n", ti, to, to * IN + ti, in[to * IN + ti]);
			if(to+start_out<OUT&&ti+start_in<IN)
			out[to][ti] = in[offset_weight+(to+start_out) * IN + ti+start_in];
			else out[to][ti] = 0;
		}
	}
}
float mac16(float f1,float w1,
		float f2,float w2,
		float f3,float w3,
		float f4,float w4,
		float f5,float w5,
		float f6,float w6,
		float f7,float w7,
		float f8,float w8,
		float f9,float w9,
		float f10,float w10,
		float f11,float w11,
		float f12,float w12,
		float f13,float w13,
		float f14,float w14,
		float f15,float w15,
		float f16,float w16)
{
#pragma HLS INLINE off
	float m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16;
	float sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8;
	float sum9,sum10,sum11,sum12,sum13,sum14;
	m1=f1*w1;
	m2=f2*w2;
	m3=f3*w3;
	m4=f4*w4;
	m5=f5*w5;
	m6=f6*w6;
	m7=f7*w7;
	m8=f8*w8;
	m9=f9*w9;
	m10=f10*w10;
	m11=f11*w11;
	m12=f12*w12;
	m13=f13*w13;
	m14=f14*w14;
	m15=f15*w15;
	m16=f16*w16;

	sum1=m1+m2;
	sum2=m3+m4;
	sum3=m5+m6;
	sum4=m7+m8;
	sum5=m9+m10;
	sum6=m11+m12;
	sum7=m13+m14;
	sum8=m15+m16;

	sum9=sum1+sum2;
	sum10=sum3+sum4;
	sum11=sum5+sum6;
	sum12=sum7+sum8;

	sum13=sum9+sum10;
	sum14=sum11+sum12;

	return sum13+sum14;
}
void load_w1x1(float wbuf1x1[CONV1x1_Tout][CONV1x1_Tin],float W1x1[CONV1x1_Tout][16],int CI)
{
#pragma HLS INLINE off
	for(int ci=0;ci<16;ci++)
	{
#pragma HLS UNROLL
		for(int co=0;co<CONV1x1_Tout;co++)
		{
#pragma HLS UNROLL
			W1x1[co][ci]=wbuf1x1[co][ci+CI];
		}
	}
}
void basic_conv_1x1(float in[CONV1x1_Tin][CONV1x1_Tw][CONV1x1_Th], float weight[CONV1x1_Tout][CONV1x1_Tin], float out[CONV1x1_Tout][CONV1x1_Tw][CONV1x1_Th])
{
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
	float W1x1[CONV1x1_Tout][16];
#pragma HLS ARRAY_PARTITION variable=W1x1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W1x1 complete dim=2

	for (int ti = 0; ti < CONV1x1_Tin; ti+=16)
	{
		load_w1x1(weight,W1x1,ti);
		for (int tw = 0; tw < CONV1x1_Tw; tw++)
		{
			for (int th = 0; th < CONV1x1_Th; th++)
			{
#pragma HLS PIPELINE II=8
				for (int to = 0; to < CONV1x1_Tout; to++)
				{
#pragma HLS UNROLL
					float tp=out[to][tw][th];
					tp += mac16(in[ti+0][tw][th],W1x1[to][0],
							in[ti+1][tw][th],W1x1[to][1],
							in[ti+2][tw][th],W1x1[to][2],
							in[ti+3][tw][th],W1x1[to][3],
							in[ti+4][tw][th],W1x1[to][4],
							in[ti+5][tw][th],W1x1[to][5],
							in[ti+6][tw][th],W1x1[to][6],
							in[ti+7][tw][th],W1x1[to][7],
							in[ti+8][tw][th],W1x1[to][8],
							in[ti+9][tw][th],W1x1[to][9],
							in[ti+10][tw][th],W1x1[to][10],
							in[ti+11][tw][th],W1x1[to][11],
							in[ti+12][tw][th],W1x1[to][12],
							in[ti+13][tw][th],W1x1[to][13],
							in[ti+14][tw][th],W1x1[to][14],
							in[ti+15][tw][th],W1x1[to][15]
							);
					out[to][tw][th]=tp;
				}

			}
		}
	}
}
void back_result_1x1(float in[CONV1x1_Tin][CONV1x1_Tw][CONV1x1_Th], float *out, int cout, int row, int col,unsigned OUT,unsigned W,unsigned H)
{
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
	int start_out = cout;
	int start_w = row;
	int start_h = col;
	for (int tw = 0; tw < 0 + POOL_Tw; tw++)
	{
		for (int th = 0; th < 0 + POOL_Th; th++)
		{
#pragma HLS PIPELINE
			for (int to = 0; to < 0 + POOL_Tout; to++)
			{

					if(tw +start_w< W && th+start_h < H && to+start_out < OUT)
					out[(to+start_out) * W * H + (tw+start_w) * H + th+start_h] =
						in[to][tw][th];

			}
		}
	}
}

template<unsigned w,unsigned h,unsigned cin,unsigned cout>
void conv_1x1(float *in,float *weight,float *out)
{
	float temp_weight1[CONV1x1_Tout][CONV1x1_Tin];
	float temp_feature1[CONV1x1_Tout][CONV1x1_Tw][CONV1x1_Th];
	float temp_feature2[CONV1x1_Tout][CONV1x1_Tw][CONV1x1_Th];
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
				clean<CONV1x1_Tout,CONV1x1_Tw,CONV1x1_Th>(temp_feature2);
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight_1x1(weight, temp_weight1, cou*CONV1x1_Tout, ci*CONV1x1_Tin,cout,cin);
					get_block_feature_1x1(in, temp_feature1,  co*CONV1x1_Th, ro*CONV1x1_Tw, ci * CONV1x1_Tin,cin,w,h);
					basic_conv_1x1(temp_feature1, temp_weight1, temp_feature2);
				}
				back_result_1x1(temp_feature2, out, cou * CONV1x1_Tout, ro * CONV1x1_Tw, co * CONV1x1_Th,cout,w,h);
			}
		}
	}
}
//BN
void basic_batchnorm(float in[BN_Tin][BN_Tw][BN_Th], float out[BN_Tin][BN_Tw][BN_Th], float gamma[BN_Tin], float beta[BN_Tin])
{
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1


	for (int tw = 0;tw < BN_Tw;tw++)
	{
		for (int th = 0;th < BN_Th;th++)
		{
#pragma HLS PIPELINE II=4
			for (int ti = 0;ti < BN_Tin;ti++)
			{
#pragma HLS UNROLL
				out[ti][tw][th] = gamma[ti] * in[ti][tw][th] + beta[ti];
				if(out[ti][tw][th]<0) out[ti][tw][th] =0;
				in[ti][tw][th]=0;//ÇåÁã
			}
		}
	}

}
void get_block_weight_BN(float* in_g, float* in_b,float gamma[BN_Tin], float beta[BN_Tin],int cin,unsigned IN)
{
	for (int i = 0;i < BN_Tin;i++)
	{
		if (i+cin < IN)
		{
			gamma[i] = in_g[offset_bias+i+cin];
			beta [i] = in_b[offset_bias+i+cin];

		}
		else
		{
			gamma[i] = 0;
			beta [i] = 0;
		}
	}
}
void back_result_BN(float in[BN_Tin][BN_Tw][BN_Th], float* out, int cout, int row, int col,unsigned OUT, unsigned W, unsigned H)
{
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
	for (int tw = 0;tw < 0 + BN_Tw;tw++)
	{
		for (int th = 0;th < 0 + BN_Th;th++)
		{
#pragma HLS PIPELINE
			for (int to = 0;to < 0 + BN_Tin;to++)
			{
				if (tw+row < W-2 && th+col < H-2 && to+cout < OUT)
					out[(to+cout) * W * H + (tw+row+1) * H + th+col+1] = in[to][tw][th];
			}
		}
	}
}
//in 3*320*160
//weight
//16*3*3*3+
//32*16*3*3+
//64*32*3*3+
//64*64*3*3(x5)+
//60*64*1*1
//gamma beta var mean 16+32+64*6
//temp 32*160*160
static layer config[32] = {
{ "padding1", 320,160,3 , 322,162,3 , 0 },  //pad1
{ "conv1",    322,162,3 , 320,160,16, 3 },  //conv1
{ "bn1",      320,160,16, 320,160,16, 0 },  //bn1
{ "pool1",    320,160,16, 160,80 ,16, 2 },  //pool1

{ "padding2", 160,80 ,16, 162,82 ,16, 0 },  //pad2
{ "conv2",    162,82 ,16, 160,80 ,32, 3 },  //conv2
{ "bn2",      160,80 ,32, 160,80 ,32, 0 },  //bn2
{ "pool2",    160,80 ,32, 80 ,40 ,32, 2 },  //pool2

{ "padding3", 80 ,40 ,32, 82 ,42 ,32, 0 },  //pad3
{ "conv3",    82 ,42 ,32, 80 ,40 ,64, 3 },  //conv3
{ "bn3",      80 ,40 ,64, 80 ,40 ,64, 0 },  //bn3
{ "pool3",    80 ,40 ,64, 40 ,20 ,64, 2 },  //pool3

{ "padding4", 40 ,20 ,64, 42 ,22 ,64, 0 },  //pad4
{ "conv4",    42 ,22 ,64, 40 ,20 ,64, 3 },  //conv4
{ "bn4",      40 ,20 ,64, 40 ,20 ,64, 0 },  //bn4
{ "pool4",    40 ,20 ,64, 20 ,10 ,64, 2 },  //pool4

{ "padding5", 20 ,10 ,64, 22 ,12 ,64, 0 },  //pad5
{ "conv5",    22 ,12 ,64, 20 ,10 ,64, 3 },  //conv5
{ "bn5",      20 ,10 ,64, 20 ,10 ,64, 0 },  //bn5

{ "padding6", 20 ,10 ,64, 22 ,12 ,64, 0 },  //pad6
{ "conv6",    22 ,12 ,64, 20 ,10 ,64, 3 },  //conv6
{ "bn6",      20 ,10 ,64, 20 ,10 ,64, 0 },  //bn6

{ "padding7", 20 ,10 ,64, 22 ,12 ,64, 0 },  //pad7
{ "conv7",    22 ,12 ,64, 20 ,10 ,64, 3 },  //conv7
{ "bn7",      20 ,10 ,64, 20 ,10 ,64, 0 },  //bn7

{ "padding8", 20 ,10 ,64, 22 ,12 ,64, 0 },  //pad8
{ "conv8",    22 ,12 ,64, 20 ,10 ,64, 3 },  //conv8
{ "bn8",      20 ,10 ,64, 20 ,10 ,64, 0 },  //bn8

{ "padding9", 20 ,10 ,64, 22 ,12 ,64, 0 },  //pad9
{ "conv9",    22 ,12 ,64, 20 ,10 ,64, 3 },  //conv9
{ "bn9",      20 ,10 ,64, 20 ,10 ,64, 0 },  //bn9

{ "conv10",   20 ,10 ,64, 20 ,10 ,60, 1 },  //conv10
};
//
void top2(float in[156492],float weight[211632],float gamma[432],float beta[432],float temp1[409600],float temp2[409600])
{
#pragma HLS ALLOCATION instances=basic_conv3x3 limit=1 function
#pragma HLS ALLOCATION instances=basic_conv_1x1 limit=1 function
#pragma HLS ALLOCATION instances=basic_batchnorm limit=1 function
#pragma HLS ALLOCATION instances=basic_maxpool limit=1 function

#pragma HLS ALLOCATION instances=get_block_weight3x3 limit=1 function
#pragma HLS ALLOCATION instances=get_block_feature3x3 limit=1 function
#pragma HLS ALLOCATION instances=add_block3x3 limit=1 function
#pragma HLS ALLOCATION instances=get_block_weight_BN limit=1 function

#pragma HLS ALLOCATION instances=back_result_POOL limit=1 function
#pragma HLS ALLOCATION instances=back_result_BN limit=1 function


#pragma HLS INTERFACE m_axi depth=4294967295 port=temp1 offset=slave bundle=TEMP1
#pragma HLS INTERFACE m_axi depth=4294967295 port=temp2 offset=slave bundle=TEMP2

#pragma HLS INTERFACE m_axi depth=4294967295 port=gamma offset=slave bundle=GAMMA
#pragma HLS INTERFACE m_axi depth=4294967295 port=beta offset=slave bundle=BETA
#pragma HLS INTERFACE m_axi depth=4294967295 port=weight offset=slave bundle=WEIGHT
#pragma HLS INTERFACE m_axi depth=153600 port=in offset=slave bundle=DATA_IN
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL

	float temp_weight1[CONV_Tout][CONV_Tin][K][K];
	float temp_weight2[CONV1x1_Tout][CONV1x1_Tin];
	float temp_feature1[CONV_Tin][CONV_Tw][CONV_Th];
	float temp_feature2[CONV_Tin][CONV_Tw-2][CONV_Th-2];

	float temp_feature4[BN_Tin][BN_Tw][BN_Th];

	float temp_feature5[POOL_Tout][POOL_Tw/2][POOL_Th/2];

	float temp_feature6[CONV1x1_Tout][CONV1x1_Tw][CONV1x1_Th];
	float temp_feature7[CONV1x1_Tout][CONV1x1_Tw][CONV1x1_Th];

	float temp_gamma[BN_Tin];
	float temp_beta[BN_Tin];

	int row_num = 0;
	int col_num = 0;
	int cin_num = 0;
	int cout_num = 0;
	int w=0,h=0,cin=0,cout=0;

//block1
	clean_temp<16,162,82>(temp2);
	offset_weight=0;
	offset_bias=0;
	w=322;
	h=162;
	cin=3;
	cout=16;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				//clean<CONV_Tin,(CONV_Tw-2),(CONV_Th-2)>(temp_feature2);
				for (int ci = 0; ci < cin_num; ci++)
				{

					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}

				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				basic_maxpool(temp_feature4,temp_feature5);
				back_result_POOL(temp_feature5, temp2, cou * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2),cout,(w-2)/2+2,(h-2)/2+2);
			}
		}
	}

//block2
	clean_temp<32,82,42>(temp1);
	offset_weight+=3*16*3*3;
	offset_bias+=16;
	w=162;
	h=82;
	cin=16;
	cout=32;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{

				for (int ci = 0; ci < cin_num; ci++)
				{

					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(temp2, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}

				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				basic_maxpool(temp_feature4,temp_feature5);
				back_result_POOL(temp_feature5, temp1, cou * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2),cout,(w-2)/2+2,(h-2)/2+2);
			}
		}
	}
//block3
	clean_temp<64,42,22>(temp2);
	offset_weight+=16*32*3*3;
	offset_bias+=32;

	w=82;
	h=42;
	cin=32;
	cout=64;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{

					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(temp1, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				basic_maxpool(temp_feature4,temp_feature5);
				back_result_POOL(temp_feature5, temp2, cou * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2),cout,(w-2)/2+2,(h-2)/2+2);
			}
		}
	}

//block4
	clean_temp<64,22,12>(temp1);
	offset_weight+=32*64*3*3;
	offset_bias+=64;
	w=42;
	h=22;
	cin=64;
	cout=64;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(temp2, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}

				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				basic_maxpool(temp_feature4,temp_feature5);
				back_result_POOL(temp_feature5, temp1, cou * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2),cout,(w-2)/2+2,(h-2)/2+2);
			}
		}
	}

//block5
	clean_temp<64,22,12>(temp2);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(temp1, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,temp2, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}

//block6
	clean_temp<64,22,12>(temp1);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(temp2, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,temp1, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}
//block7
	clean_temp<64,22,12>(temp2);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(temp1, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,temp2, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}
//block8
	clean_temp<64,22,12>(temp1);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(temp2, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,temp1, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}

	//block9
	clean_temp<64,22,12>(temp2);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(weight, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(temp1, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(gamma,beta,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,temp2, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}
//block10
	offset_weight+=64*64*3*3;
	w=20;
	h=10;
	cin=64;
	cout=60;
	if(w%(CONV_Tw - 2)!=0) row_num = w / (CONV_Tw - 2)+1 ;
		else row_num = w / (CONV_Tw - 2);
	if(h%(CONV_Th - 2)!=0) col_num = h / (CONV_Th - 2)+1 ;
		else col_num = h / (CONV_Th - 2);
	if(cin % CONV_Tin!=0)cin_num=cin / CONV_Tin+1;
		else cin_num=cin / CONV_Tin;
	if(cout % CONV_Tout!=0)cout_num=cout / CONV_Tout+1;
		else cout_num=cout / CONV_Tout;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				clean<CONV1x1_Tout,CONV1x1_Tw,CONV1x1_Th>(temp_feature7);
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight_1x1(weight, temp_weight2, cou*CONV1x1_Tout, ci*CONV1x1_Tin,cout,cin);
					get_block_feature_1x1(temp2, temp_feature6,  co*CONV1x1_Th, ro*CONV1x1_Tw, ci * CONV1x1_Tin,cin,w,h);
					basic_conv_1x1(temp_feature6, temp_weight2, temp_feature7);
				}
				back_result_1x1(temp_feature7, temp1, cou * CONV1x1_Tout, ro * CONV1x1_Tw, co * CONV1x1_Th,cout,w,h);
			}
		}
	}

}

void top(float in[1219164])
{
#pragma HLS ALLOCATION instances=basic_conv3x3 limit=1 function
#pragma HLS ALLOCATION instances=basic_conv_1x1 limit=1 function
#pragma HLS ALLOCATION instances=basic_batchnorm limit=1 function
#pragma HLS ALLOCATION instances=basic_maxpool limit=1 function

#pragma HLS ALLOCATION instances=get_block_weight3x3 limit=1 function
#pragma HLS ALLOCATION instances=get_block_feature3x3 limit=1 function
#pragma HLS ALLOCATION instances=add_block3x3 limit=1 function
#pragma HLS ALLOCATION instances=get_block_weight_BN limit=1 function

#pragma HLS ALLOCATION instances=back_result_POOL limit=1 function
#pragma HLS ALLOCATION instances=back_result_BN limit=1 function


#pragma HLS INTERFACE m_axi depth=1219164*4 port=in offset=slave bundle=DATA_IN
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL
	int off_w= 156492;
	int off_g= 156492+211632;
	int off_b= 156492+211632+432;
	int off_t1=156492+211632+432+432;
	int off_t2=156492+211632+432+432+425088;
	float temp_weight1[CONV_Tout][CONV_Tin][K][K];
	float temp_weight2[CONV1x1_Tout][CONV1x1_Tin];
	float temp_feature1[CONV_Tin][CONV_Tw][CONV_Th];
	float temp_feature2[CONV_Tin][CONV_Tw-2][CONV_Th-2]={0};

	float temp_feature4[BN_Tin][BN_Tw][BN_Th];

	float temp_feature5[POOL_Tout][POOL_Tw/2][POOL_Th/2];

	float temp_feature6[CONV1x1_Tout][CONV1x1_Tw][CONV1x1_Th];
	float temp_feature7[CONV1x1_Tout][CONV1x1_Tw][CONV1x1_Th];

	float temp_gamma[BN_Tin];
	float temp_beta[BN_Tin];

	int row_num = 0;
	int col_num = 0;
	int cin_num = 0;
	int cout_num = 0;
	int w=0,h=0,cin=0,cout=0;

//block1
	clean_temp<16,162,82>(in+off_t2);
	offset_weight=0;
	offset_bias=0;
	w=322;
	h=162;
	cin=3;
	cout=16;
	row_num =(w-3)/(CONV_Tw-2)+1;
	col_num =(h-3)/(CONV_Th-2)+1;
	cin_num =(cin-1) / CONV_Tin+1;
	cout_num=(cout-1) / CONV_Tout+1;
int flag=0;
	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				//clean<CONV_Tin,(CONV_Tw-2),(CONV_Th-2)>(temp_feature2);
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(in+off_w, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
//					for(int i=1;i<2;i++)
//					{
//						for(int j=0;j<22;j++)
//						{
//							for(int k=0;k<22;k++)
//							{
//								std::cout<<temp_feature1[i][k][j]<<" ";
//							}
//							std::cout<<std::endl;
//						}
//						std::cout<<std::endl;
//					}std::cout<<std::endl;
				}
				get_block_weight_BN(in+off_g,in+off_b,temp_gamma,temp_beta,cou*BN_Tin,cout);
//				if(co==1&&cou==0&&ro==0)
//				{std::cout<<"conv1!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
//				for(int i=1;i<2;i++)
//				{
//					for(int k=0;k<22;k++)
//					{
//						for(int j=0;j<22;j++)
//						{
//							std::cout<<temp_feature1[0][j][k]<<" ";
//						}
//						std::cout<<std::endl;
//					}std::cout<<std::endl;
//				}
//				for(int i=1;i<2;i++)
//				{
//					for(int k=0;k<20;k++)
//					{
//						for(int j=0;j<20;j++)
//						{
//							std::cout<<temp_feature2[0][j][k]<<" ";
//						}
//						std::cout<<std::endl;
//					}
//					std::cout<<std::endl;
//				}std::cout<<std::endl;
//				}
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
//				//if(co==1&&cou==0&&ro==0)
//								{std::cout<<"bn1!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
//								//std::cout<<temp_gamma[1]<<" "<<temp_beta[1]<<std::endl;
//								for(int i=0;i<1;i++)
//								{
//									for(int k=0;k<20;k++)
//									{
//										for(int j=0;j<20;j++)
//										{
//											std::cout<<temp_feature4[0][j][k]<<" ";
//										}
//										std::cout<<std::endl;
//									}
//									std::cout<<std::endl;
//								}std::cout<<std::endl;
//								}
				basic_maxpool(temp_feature4,temp_feature5);
//				if(co==1&&cou==0&&ro==0)
//								{std::cout<<"pool1!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
//								for(int i=0;i<1;i++)
//								{
//									for(int k=0;k<10;k++)
//									{
//										for(int j=0;j<10;j++)
//										{
//											std::cout<<temp_feature5[1][k][j]<<" ";
//										}
//										std::cout<<std::endl;
//									}
//									std::cout<<std::endl;
//								}std::cout<<std::endl;
//								}
//				flag=1;
				back_result_POOL(temp_feature5, in+off_t2, cou * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2),cout,(w-2)/2+2,(h-2)/2+2);
			}
		}
	}

//block2
	clean_temp<32,82,42>(in+off_t1);
	offset_weight+=3*16*3*3;
	offset_bias+=16;
	w=162;
	h=82;
	cin=16;
	cout=32;
	row_num =(w-3)/(CONV_Tw-2)+1;
	col_num =(h-3)/(CONV_Th-2)+1;
	cin_num =(cin-1) / CONV_Tin+1;
	cout_num=(cout-1) / CONV_Tout+1;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{

				for (int ci = 0; ci < cin_num; ci++)
				{

					get_block_weight3x3(in+off_w, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in+off_t2, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}

				get_block_weight_BN(in+off_g,in+off_b,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				basic_maxpool(temp_feature4,temp_feature5);
				back_result_POOL(temp_feature5, in+off_t1, cou * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2),cout,(w-2)/2+2,(h-2)/2+2);
			}
		}
	}

//block3
	clean_temp<64,42,22>(in+off_t2);
	offset_weight+=16*32*3*3;
	offset_bias+=32;

	w=82;
	h=42;
	cin=32;
	cout=64;
	row_num =(w-3)/(CONV_Tw-2)+1;
	col_num =(h-3)/(CONV_Th-2)+1;
	cin_num =(cin-1) / CONV_Tin+1;
	cout_num=(cout-1) / CONV_Tout+1;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{

					get_block_weight3x3(in+off_w, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in+off_t1, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}

				get_block_weight_BN(in+off_g,in+off_b,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);

				basic_maxpool(temp_feature4,temp_feature5);
				back_result_POOL(temp_feature5, in+off_t2, cou * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2),cout,(w-2)/2+2,(h-2)/2+2);
			}
		}
	}

//block4
	clean_temp<64,22,12>(in+off_t1);
	offset_weight+=32*64*3*3;
	offset_bias+=64;
	w=42;
	h=22;
	cin=64;
	cout=64;
	row_num =(w-3)/(CONV_Tw-2)+1;
	col_num =(h-3)/(CONV_Th-2)+1;
	cin_num =(cin-1) / CONV_Tin+1;
	cout_num=(cout-1) / CONV_Tout+1;
	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(in+off_w, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in+off_t2, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}

				get_block_weight_BN(in+off_g,in+off_b,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				basic_maxpool(temp_feature4,temp_feature5);
				back_result_POOL(temp_feature5, in+off_t1, cou * (POOL_Tout), ro * (POOL_Tw/2), co * (POOL_Th/2),cout,(w-2)/2+2,(h-2)/2+2);
			}
		}
	}

//block5
	clean_temp<64,22,12>(in+off_t2);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	row_num =(w-3)/(CONV_Tw-2)+1;
	col_num =(h-3)/(CONV_Th-2)+1;
	cin_num =(cin-1) / CONV_Tin+1;
	cout_num=(cout-1) / CONV_Tout+1;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(in+off_w, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in+off_t1, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(in+off_g,in+off_b,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,in+off_t2, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}

//block6
	clean_temp<64,22,12>(in+off_t1);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	row_num =(w-3)/(CONV_Tw-2)+1;
	col_num =(h-3)/(CONV_Th-2)+1;
	cin_num =(cin-1) / CONV_Tin+1;
	cout_num=(cout-1) / CONV_Tout+1;
	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(in+off_w, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in+off_t2, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(in+off_g,in+off_b,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,in+off_t1, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}
//block7
	clean_temp<64,22,12>(in+off_t2);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	row_num =(w-3)/(CONV_Tw-2)+1;
	col_num =(h-3)/(CONV_Th-2)+1;
	cin_num =(cin-1) / CONV_Tin+1;
	cout_num=(cout-1) / CONV_Tout+1;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(in+off_w, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in+off_t1, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);
					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(in+off_g,in+off_b,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,in+off_t2, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}

//block8
	clean_temp<64,22,12>(in+off_t1);
	offset_weight+=64*64*3*3;
	offset_bias+=64;
	w=22;
	h=12;
	cin=64;
	cout=64;
	row_num =(w-3)/(CONV_Tw-2)+1;
	col_num =(h-3)/(CONV_Th-2)+1;
	cin_num =(cin-1) / CONV_Tin+1;
	cout_num=(cout-1) / CONV_Tout+1;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight3x3(in+off_w, temp_weight1, cou*CONV_Tout, ci*CONV_Tin,cout,cin,3);
					get_block_feature3x3(in+off_t2, temp_feature1,ci * CONV_Tin, ro*(CONV_Tw-2),co*(CONV_Th-2),w,h,cin);

					basic_conv3x3(temp_feature1, temp_weight1, temp_feature2);
				}
				get_block_weight_BN(in+off_g,in+off_b,temp_gamma,temp_beta,cou*BN_Tin,cout);
				basic_batchnorm(temp_feature2, temp_feature4, temp_gamma, temp_beta);
				back_result_BN(temp_feature4,in+off_t1, cou * BN_Tin, ro * BN_Tw, co * BN_Th,cout,w,h);
			}
		}
	}

//block9
	offset_weight+=64*64*3*3;
	w=20;
	h=10;
	cin=64;
	cout=60;
	row_num =(w-1)/CONV1x1_Tw+1;
	col_num =(h-1)/CONV1x1_Th+1;
	cin_num =(cin-1) / CONV1x1_Tin+1;
	cout_num=(cout-1) / CONV1x1_Tout+1;

	for(int ro=0;ro<row_num;ro++)
	{
		for (int co = 0; co < col_num; co++)
		{
			for (int cou = 0; cou < cout_num; cou++)
			{
				clean<CONV1x1_Tout,CONV1x1_Tw,CONV1x1_Th>(temp_feature7);
				for (int ci = 0; ci < cin_num; ci++)
				{
					get_block_weight_1x1(in+off_w, temp_weight2, cou*CONV1x1_Tout, ci*CONV1x1_Tin,cout,cin);
					get_block_feature_1x1(in+off_t1, temp_feature6,  co*CONV1x1_Th, ro*CONV1x1_Tw, ci * CONV1x1_Tin,cin,w,h);
					basic_conv_1x1(temp_feature6, temp_weight2, temp_feature7);
				}
				back_result_1x1(temp_feature7, in+off_t2, cou * CONV1x1_Tout, ro * CONV1x1_Tw, co * CONV1x1_Th,cout,w,h);
			}
		}
	}
}
