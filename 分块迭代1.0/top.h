#define BATCH 1

#define CONV_Tin 2
#define CONV_Tout 2
#define CONV_Tw 8
#define CONV_Th 8
#define K 3

#define CONV1x1_Tin 1
#define CONV1x1_Tout 1
#define CONV1x1_Tw 28
#define CONV1x1_Th 28

#define POOL_Tout 16
#define POOL_Tw 4
#define POOL_Th 4

#define BN_Tin 3
#define BN_Tw 5
#define BN_Th 5
#define eps 1e-05
template<unsigned w_in,unsigned w_out,unsigned w_k>
void load_conv_weight(float *in, float* out,int offset);
template<unsigned w_in>
void load_conv_bias(float *in, float* out,int offset);
template<unsigned f_W,unsigned f_H,unsigned f_in_ch,unsigned value>
void padding(float *in, float *out);
template<unsigned a>
void clean(float *in);

//conv3x3
template<unsigned W,unsigned H,unsigned IN>
void get_block_feature3x3(float *in, float *out,int cin,int row,int col);
template<unsigned OUT,unsigned IN,unsigned k>
void get_block_weight3x3(float *in, float *out, int cout, int cin);
template<unsigned W,unsigned H,unsigned OUT>
void back_result3x3(float *in, float *out, int cout, int row, int col);
void basic_conv3x3(float in[CONV_Tin][CONV_Tw][CONV_Th], float weight[CONV_Tout][CONV_Tin][K][K], float out[CONV_Tout][CONV_Tw][CONV_Th]);
template<unsigned w,unsigned h,unsigned cout,unsigned cin>
void conv3x3_block(float *in,float *weight,float *out);

//pool
template<int IN,int W,int H>
void get_block_feature_POOL(float *in, float *out, int col, int row,int cin);
template<unsigned OUT>
void get_block_bias_POOL(float *in[128], float *out, int cout);
template<unsigned OUT,unsigned W,unsigned H>
void back_result_POOL(float *in, float *out, int cout, int row, int col);
void basic_bias_relu_maxpool(float in[BATCH*POOL_Tout*POOL_Tw*POOL_Th],  float bias[POOL_Tout], float out[BATCH*POOL_Tout*(POOL_Tw/2)*(POOL_Th/2)]);
template<unsigned w,unsigned h,unsigned cout>
void bias_relu_pool(float *in,float *bias,float *out);

//conv1X1
template<unsigned IN,unsigned W,unsigned H>
void get_block_feature_1x1(float *in, float *out, int col, int row,int cin);
template<unsigned OUT, unsigned IN >
void get_block_weight_1x1(float *in, float *out, int cout, int cin);
void basic_conv_1x1(float in[CONV1x1_Tin*CONV1x1_Tw*CONV1x1_Th], float weight[CONV1x1_Tout*CONV1x1_Tin*1*1], float out[CONV1x1_Tout*CONV1x1_Tw*CONV1x1_Th]);
template<unsigned OUT,unsigned W,unsigned H>
void back_result_1x1(float *in, float *out, int cout, int row, int col);
template<unsigned w,unsigned h,unsigned cin,unsigned cout>
void conv_1x1(float *in,float *weight,float *out);

//BN
void basic_batchnorm(float in[BATCH * BN_Tin * BN_Tw * BN_Th], float out[BATCH * BN_Tin * BN_Tw * BN_Th], float gamma[BN_Tin], float beta[BN_Tin], float mean[BN_Tin], float var[BN_Tin]);
template<unsigned IN,unsigned W,unsigned H>
void get_block_feature_BN(float *in,float *out,int col,int row,int cin);
template<unsigned IN>
void get_block_weight_BN(float* in_g, float* in_b, float* in_m, float* in_v, float gamma[BN_Tin], float beta[BN_Tin], float mean[BN_Tin], float var[BN_Tin],int cin);
template<unsigned OUT, unsigned W, unsigned H>
void back_result_BN(float *in, float* out, int cout, int row, int col);
template<unsigned w,unsigned h,unsigned cout>
void batchnorm(float* in, float* out, float* gamma, float* beta, float* mean, float* var);




template<unsigned f_W,unsigned f_H,unsigned w_k,unsigned f_in_ch,unsigned type>
void pool(float *in, float *out);

template<unsigned f_W,unsigned f_H,unsigned w_k,unsigned f_out_ch,unsigned f_in_ch>
void conv(float *in, float *weight,float *bias, float *out);

void top(float in[1*28*28],float weight[206736],float bias[186],float out[10],float temp1[12544],float temp2[12544]);
