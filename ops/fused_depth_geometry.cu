#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__device__ inline float safe_inv_norm(float x, float y, float z, float eps) {
  float n2 = x * x + y * y + z * z;
  return n2 > eps * eps ? 1.0f / sqrtf(n2) : 0.0f;
}

__global__ void fused_kernel(
    const float* __restrict__ provisional,
    const float* __restrict__ rest,
    const float* __restrict__ target,
    const bool* __restrict__ has_direction,
    const float* __restrict__ confidence,
    float* __restrict__ source_out,
    float* __restrict__ target_out,
    float* __restrict__ disagreement_out,
    int n, int t_count, int k_count, float eps) {
  int nt = blockIdx.x;
  if (nt >= n * t_count) return;
  int n_idx = nt / t_count;
  int t_idx = nt - n_idx * t_count;
  int k = threadIdx.x;

  __shared__ float angle_sum[32];
  __shared__ float conf_sum[32];
  float weighted = 0.0f;
  float weight = 0.0f;
  if (k < k_count) {
    const float* m = provisional + ((n_idx * t_count + t_idx) * 9);
    const float* r = rest + (n_idx * k_count + k) * 3;
    float sx = m[0] * r[0] + m[1] * r[1] + m[2] * r[2];
    float sy = m[3] * r[0] + m[4] * r[1] + m[5] * r[2];
    float sz = m[6] * r[0] + m[7] * r[1] + m[8] * r[2];
    float inv = safe_inv_norm(sx, sy, sz, eps);
    if (inv > 0.0f) { sx *= inv; sy *= inv; sz *= inv; }
    else { sx = 1.0f; sy = 0.0f; sz = 0.0f; }
    int out_idx = ((n_idx * t_count + t_idx) * k_count + k) * 3;
    source_out[out_idx + 0] = sx;
    source_out[out_idx + 1] = sy;
    source_out[out_idx + 2] = sz;
    int dir_idx = (n_idx * t_count + t_idx) * k_count + k;
    float tx = target[dir_idx * 3 + 0];
    float ty = target[dir_idx * 3 + 1];
    float tz = target[dir_idx * 3 + 2];
    if (!has_direction[dir_idx]) { tx = sx; ty = sy; tz = sz; }
    target_out[out_idx + 0] = tx;
    target_out[out_idx + 1] = ty;
    target_out[out_idx + 2] = tz;
    float dot = fminf(1.0f, fmaxf(-1.0f, sx * tx + sy * ty + sz * tz));
    float cx = sy * tz - sz * ty;
    float cy = sz * tx - sx * tz;
    float cz = sx * ty - sy * tx;
    float angle = atan2f(sqrtf(cx * cx + cy * cy + cz * cz), dot);
    float c = confidence[dir_idx];
    c = isfinite(c) ? fmaxf(c, 0.0f) : 0.0f;
    weighted = angle * c;
    weight = c;
  }
  angle_sum[k] = weighted;
  conf_sum[k] = weight;
  __syncthreads();
  for (int stride = 16; stride > 0; stride >>= 1) {
    if (k < stride) {
      angle_sum[k] += angle_sum[k + stride];
      conf_sum[k] += conf_sum[k + stride];
    }
    __syncthreads();
  }
  if (k == 0) {
    disagreement_out[nt] = angle_sum[0] / fmaxf(conf_sum[0], eps);
  }
}

__global__ void axis_angle_matrix_kernel(
    const float* __restrict__ vector,
    float* __restrict__ matrix,
    int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float x = vector[index * 3 + 0];
  float y = vector[index * 3 + 1];
  float z = vector[index * 3 + 2];
  float theta2 = x * x + y * y + z * z;
  float a, b;
  if (theta2 < 1.0e-8f) {
    // Stable Taylor evaluation of the exact Rodrigues coefficients.
    float theta4 = theta2 * theta2;
    a = 1.0f - theta2 / 6.0f + theta4 / 120.0f;
    b = 0.5f - theta2 / 24.0f + theta4 / 720.0f;
  } else {
    float theta = sqrtf(theta2);
    a = sinf(theta) / theta;
    b = (1.0f - cosf(theta)) / theta2;
  }
  float* r = matrix + index * 9;
  r[0] = 1.0f - b * (y * y + z * z);
  r[1] = b * x * y - a * z;
  r[2] = b * x * z + a * y;
  r[3] = b * x * y + a * z;
  r[4] = 1.0f - b * (x * x + z * z);
  r[5] = b * y * z - a * x;
  r[6] = b * x * z - a * y;
  r[7] = b * y * z + a * x;
  r[8] = 1.0f - b * (x * x + y * y);
}

__global__ void matrix_axis_angle_kernel(
    const float* __restrict__ matrix,
    float* __restrict__ vector,
    int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float* m = matrix + index * 9;
  float qabs[4];
  qabs[0] = sqrtf(fmaxf(0.0f, 1.0f + m[0] + m[4] + m[8]));
  qabs[1] = sqrtf(fmaxf(0.0f, 1.0f + m[0] - m[4] - m[8]));
  qabs[2] = sqrtf(fmaxf(0.0f, 1.0f - m[0] + m[4] - m[8]));
  qabs[3] = sqrtf(fmaxf(0.0f, 1.0f - m[0] - m[4] + m[8]));
  int best = 0;
  if (qabs[1] > qabs[best]) best = 1;
  if (qabs[2] > qabs[best]) best = 2;
  if (qabs[3] > qabs[best]) best = 3;
  float q[4];
  if (best == 0) {
    q[0] = qabs[0] * qabs[0]; q[1] = m[7] - m[5];
    q[2] = m[2] - m[6]; q[3] = m[3] - m[1];
  } else if (best == 1) {
    q[0] = m[7] - m[5]; q[1] = qabs[1] * qabs[1];
    q[2] = m[3] + m[1]; q[3] = m[2] + m[6];
  } else if (best == 2) {
    q[0] = m[2] - m[6]; q[1] = m[3] + m[1];
    q[2] = qabs[2] * qabs[2]; q[3] = m[5] + m[7];
  } else {
    q[0] = m[3] - m[1]; q[1] = m[6] + m[2];
    q[2] = m[7] + m[5]; q[3] = qabs[3] * qabs[3];
  }
  float denom = 2.0f * fmaxf(qabs[best], 0.1f);
  q[0] /= denom; q[1] /= denom; q[2] /= denom; q[3] /= denom;
  if (q[0] < 0.0f) { q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; }
  float norm = sqrtf(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  float half = atan2f(norm, q[0]);
  float angle = 2.0f * half;
  float ratio = fabsf(angle) < 1.0e-6f
      ? 0.5f - angle * angle / 48.0f
      : sinf(half) / angle;
  vector[index * 3 + 0] = q[1] / ratio;
  vector[index * 3 + 1] = q[2] / ratio;
  vector[index * 3 + 2] = q[3] / ratio;
}

__global__ void minimal_axis_angle_kernel(
    const float* __restrict__ source,
    const float* __restrict__ target,
    float* __restrict__ vector,
    int count, float eps) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float sx = source[index * 3 + 0];
  float sy = source[index * 3 + 1];
  float sz = source[index * 3 + 2];
  float tx = target[index * 3 + 0];
  float ty = target[index * 3 + 1];
  float tz = target[index * 3 + 2];
  float si = safe_inv_norm(sx, sy, sz, eps);
  float ti = safe_inv_norm(tx, ty, tz, eps);
  if (si > 0.0f) { sx *= si; sy *= si; sz *= si; }
  else { sx = 1.0f; sy = 0.0f; sz = 0.0f; }
  if (ti > 0.0f) { tx *= ti; ty *= ti; tz *= ti; }
  else { tx = 1.0f; ty = 0.0f; tz = 0.0f; }
  float cx = sy * tz - sz * ty;
  float cy = sz * tx - sx * tz;
  float cz = sx * ty - sy * tx;
  float sine = sqrtf(cx * cx + cy * cy + cz * cz);
  float dot = fminf(1.0f, fmaxf(-1.0f, sx * tx + sy * ty + sz * tz));
  float ax = cx / fmaxf(sine, eps);
  float ay = cy / fmaxf(sine, eps);
  float az = cz / fmaxf(sine, eps);
  bool anti = sine <= eps && dot < 0.0f;
  if (anti) {
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    float asx = fabsf(sx), asy = fabsf(sy), asz = fabsf(sz);
    if (asx <= asy && asx <= asz) ux = 1.0f;
    else if (asy <= asz) uy = 1.0f;
    else uz = 1.0f;
    float projection = ux * sx + uy * sy + uz * sz;
    ax = ux - projection * sx;
    ay = uy - projection * sy;
    az = uz - projection * sz;
    float ai = safe_inv_norm(ax, ay, az, eps);
    if (ai > 0.0f) { ax *= ai; ay *= ai; az *= ai; }
    else { ax = 1.0f; ay = 0.0f; az = 0.0f; }
  }
  float angle = atan2f(sine, dot);
  if (sine <= eps && !anti) { ax = 0.0f; ay = 0.0f; az = 0.0f; }
  vector[index * 3 + 0] = ax * angle;
  vector[index * 3 + 1] = ay * angle;
  vector[index * 3 + 2] = az * angle;
}

__device__ inline float det3(const float* a) {
  return a[0] * (a[4] * a[8] - a[5] * a[7])
       - a[1] * (a[3] * a[8] - a[5] * a[6])
       + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

__device__ inline void top_two_symmetric_eigenvalues(
    const float* a, float* largest, float* second) {
  float q = (a[0] + a[4] + a[8]) / 3.0f;
  float a00 = a[0] - q, a11 = a[4] - q, a22 = a[8] - q;
  float p2 = a00*a00 + a11*a11 + a22*a22
           + 2.0f * (a[1]*a[1] + a[2]*a[2] + a[5]*a[5]);
  if (p2 <= 1.0e-30f) {
    float x=a[0], y=a[4], z=a[8];
    if (x < y) { float v=x; x=y; y=v; }
    if (y < z) { float v=y; y=z; z=v; }
    if (x < y) { float v=x; x=y; y=v; }
    *largest=fmaxf(x,0.0f); *second=fmaxf(y,0.0f); return;
  }
  float p = sqrtf(p2 / 6.0f);
  float b[9];
  for (int i=0;i<9;++i) b[i]=a[i]/p;
  b[0]-=q/p; b[4]-=q/p; b[8]-=q/p;
  float r = fminf(1.0f, fmaxf(-1.0f, det3(b) * 0.5f));
  float phi = acosf(r) / 3.0f;
  float e0 = q + 2.0f*p*cosf(phi);
  float e2 = q + 2.0f*p*cosf(phi + 2.0943951023931955f);
  float e1 = 3.0f*q - e0 - e2;
  if (e0 < e1) { float v=e0; e0=e1; e1=v; }
  if (e1 < e2) { float v=e1; e1=e2; e2=v; }
  if (e0 < e1) { float v=e0; e0=e1; e1=v; }
  *largest=fmaxf(e0,0.0f); *second=fmaxf(e1,0.0f);
}

__device__ inline void minimal_matrix(
    float sx,float sy,float sz,float tx,float ty,float tz,float eps,float* r) {
  float si=safe_inv_norm(sx,sy,sz,eps), ti=safe_inv_norm(tx,ty,tz,eps);
  if(si>0){sx*=si;sy*=si;sz*=si;}else{sx=1;sy=0;sz=0;}
  if(ti>0){tx*=ti;ty*=ti;tz*=ti;}else{tx=1;ty=0;tz=0;}
  float x=sy*tz-sz*ty,y=sz*tx-sx*tz,z=sx*ty-sy*tx;
  float d=fminf(1.0f,fmaxf(-1.0f,sx*tx+sy*ty+sz*tz));
  float s2=x*x+y*y+z*z;
  if(s2<=eps){
    if(d>=0){for(int i=0;i<9;++i)r[i]=(i%4==0)?1.0f:0.0f;return;}
    float ux=0,uy=0,uz=0,asx=fabsf(sx),asy=fabsf(sy),asz=fabsf(sz);
    if(asx<=asy&&asx<=asz)ux=1;else if(asy<=asz)uy=1;else uz=1;
    float proj=ux*sx+uy*sy+uz*sz;ux-=proj*sx;uy-=proj*sy;uz-=proj*sz;
    float inv=safe_inv_norm(ux,uy,uz,eps);ux*=inv;uy*=inv;uz*=inv;
    r[0]=2*ux*ux-1;r[1]=2*ux*uy;r[2]=2*ux*uz;
    r[3]=2*uy*ux;r[4]=2*uy*uy-1;r[5]=2*uy*uz;
    r[6]=2*uz*ux;r[7]=2*uz*uy;r[8]=2*uz*uz-1;return;
  }
  float k=(1.0f-d)/s2;
  r[0]=1-k*(y*y+z*z);r[1]=-z+k*x*y;r[2]=y+k*x*z;
  r[3]=z+k*x*y;r[4]=1-k*(x*x+z*z);r[5]=-x+k*y*z;
  r[6]=-y+k*x*z;r[7]=x+k*y*z;r[8]=1-k*(x*x+y*y);
}

__global__ void kabsch_kernel(
    const float* __restrict__ source,const float* __restrict__ target,
    const float* __restrict__ confidence,float* __restrict__ rotation,
    bool* __restrict__ valid,int count,int children,float eps) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=count)return;
  float h[9]={0};int positive=0,strongest=0;float strongest_w=-1,total=0;
  for(int n=0;n<children;++n){
    int wi=idx*children+n;float w=confidence[wi];w=isfinite(w)?fmaxf(w,0.0f):0.0f;
    if(w>eps)positive++;if(w>strongest_w){strongest_w=w;strongest=n;}total+=w;
    const float* sp=source+wi*3;const float* tp=target+wi*3;
    float sx=sp[0],sy=sp[1],sz=sp[2],tx=tp[0],ty=tp[1],tz=tp[2];
    float si=safe_inv_norm(sx,sy,sz,eps),ti=safe_inv_norm(tx,ty,tz,eps);
    if(si>0){sx*=si;sy*=si;sz*=si;}else{sx=1;sy=0;sz=0;}
    if(ti>0){tx*=ti;ty*=ti;tz*=ti;}else{tx=1;ty=0;tz=0;}
    h[0]+=w*tx*sx;h[1]+=w*tx*sy;h[2]+=w*tx*sz;
    h[3]+=w*ty*sx;h[4]+=w*ty*sy;h[5]+=w*ty*sz;
    h[6]+=w*tz*sx;h[7]+=w*tz*sy;h[8]+=w*tz*sz;
  }
  float ata[9];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){float v=0;for(int k=0;k<3;++k)v+=h[k*3+i]*h[k*3+j];ata[i*3+j]=v;}
  float l0,l1;top_two_symmetric_eigenvalues(ata,&l0,&l1);
  float s0=sqrtf(l0),s1=sqrtf(l1);
  float norm=0;for(int i=0;i<9;++i)norm+=h[i]*h[i];norm=sqrtf(fmaxf(norm,0.0f));
  float scale=fmaxf(norm,eps);h[0]+=4*eps*scale;h[4]+=8*eps*scale;h[8]+=16*eps*scale;
  // Davenport matrix for B=H^T; its dominant eigenvector is the Kabsch quaternion.
  float b[9]={h[0],h[3],h[6],h[1],h[4],h[7],h[2],h[5],h[8]};
  float sigma=b[0]+b[4]+b[8],a[16]={0},v[16]={0};
  float zx=b[5]-b[7],zy=b[6]-b[2],zz=b[1]-b[3];
  a[0]=sigma;a[1]=a[4]=zx;a[2]=a[8]=zy;a[3]=a[12]=zz;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)a[(i+1)*4+j+1]=b[i*3+j]+b[j*3+i]-(i==j?sigma:0);
  for(int i=0;i<4;++i)v[i*4+i]=1;
  const int pairs[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
  for(int sweep=0;sweep<20;++sweep)for(int pp=0;pp<6;++pp){int p=pairs[pp][0],q=pairs[pp][1];float apq=a[p*4+q];if(fabsf(apq)<1e-12f)continue;float tau=(a[q*4+q]-a[p*4+p])/(2*apq);float t=copysignf(1.0f,tau)/(fabsf(tau)+sqrtf(1+tau*tau));float c=1/sqrtf(1+t*t),s=t*c;for(int k=0;k<4;++k){if(k==p||k==q)continue;float akp=a[k*4+p],akq=a[k*4+q];a[k*4+p]=a[p*4+k]=c*akp-s*akq;a[k*4+q]=a[q*4+k]=s*akp+c*akq;}float app=a[p*4+p],aqq=a[q*4+q];a[p*4+p]=c*c*app-2*s*c*apq+s*s*aqq;a[q*4+q]=s*s*app+2*s*c*apq+c*c*aqq;a[p*4+q]=a[q*4+p]=0;for(int k=0;k<4;++k){float vp=v[k*4+p],vq=v[k*4+q];v[k*4+p]=c*vp-s*vq;v[k*4+q]=s*vp+c*vq;}}
  int best=0;for(int i=1;i<4;++i)if(a[i*4+i]>a[best*4+best])best=i;
  float qr=v[best],qx=v[4+best],qy=v[8+best],qz=v[12+best];float qi=1/sqrtf(qr*qr+qx*qx+qy*qy+qz*qz);qr*=qi;qx*=qi;qy*=qi;qz*=qi;
  float kr[9];kr[0]=1-2*(qy*qy+qz*qz);kr[1]=2*(qx*qy-qz*qr);kr[2]=2*(qx*qz+qy*qr);kr[3]=2*(qx*qy+qz*qr);kr[4]=1-2*(qx*qx+qz*qz);kr[5]=2*(qy*qz-qx*qr);kr[6]=2*(qx*qz-qy*qr);kr[7]=2*(qy*qz+qx*qr);kr[8]=1-2*(qx*qx+qy*qy);
  bool good=positive>=2&&s0>eps&&s1>s0*1e-4f;
  float fallback[9];int wi=idx*children+strongest;minimal_matrix(source[wi*3],source[wi*3+1],source[wi*3+2],target[wi*3],target[wi*3+1],target[wi*3+2],eps,fallback);
  float* out=rotation+idx*9;for(int i=0;i<9;++i)out[i]=total>eps?(good?kr[i]:fallback[i]):((i%4==0)?1.0f:0.0f);valid[idx]=good;
}

__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ inline void matmul3(
    const float* a, const float* b, float* out) {
  #pragma unroll
  for (int r = 0; r < 3; ++r) {
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
      out[r * 3 + c] = a[r * 3] * b[c]
          + a[r * 3 + 1] * b[3 + c]
          + a[r * 3 + 2] * b[6 + c];
    }
  }
}

__device__ inline void parent_transpose_matmul3(
    const float* parent, const float* child, float* out) {
  #pragma unroll
  for (int r = 0; r < 3; ++r) {
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
      out[r * 3 + c] = parent[r] * child[c]
          + parent[3 + r] * child[3 + c]
          + parent[6 + r] * child[6 + c];
    }
  }
}

__device__ inline void rodrigues_vector(const float* vector, float* matrix) {
  float x = vector[0], y = vector[1], z = vector[2];
  float theta2 = x * x + y * y + z * z;
  float a, b;
  if (theta2 < 1.0e-8f) {
    float theta4 = theta2 * theta2;
    a = 1.0f - theta2 / 6.0f + theta4 / 120.0f;
    b = 0.5f - theta2 / 24.0f + theta4 / 720.0f;
  } else {
    float theta = sqrtf(theta2);
    a = sinf(theta) / theta;
    b = (1.0f - cosf(theta)) / theta2;
  }
  matrix[0] = 1.0f - b * (y * y + z * z);
  matrix[1] = b * x * y - a * z;
  matrix[2] = b * x * z + a * y;
  matrix[3] = b * x * y + a * z;
  matrix[4] = 1.0f - b * (x * x + z * z);
  matrix[5] = b * y * z - a * x;
  matrix[6] = b * x * z - a * y;
  matrix[7] = b * y * z + a * x;
  matrix[8] = 1.0f - b * (x * x + y * y);
}

__device__ inline void matrix_axis_angle(const float* m, float* vector) {
  float qabs[4];
  qabs[0] = sqrtf(fmaxf(0.0f, 1.0f + m[0] + m[4] + m[8]));
  qabs[1] = sqrtf(fmaxf(0.0f, 1.0f + m[0] - m[4] - m[8]));
  qabs[2] = sqrtf(fmaxf(0.0f, 1.0f - m[0] + m[4] - m[8]));
  qabs[3] = sqrtf(fmaxf(0.0f, 1.0f - m[0] - m[4] + m[8]));
  int best = 0;
  if (qabs[1] > qabs[best]) best = 1;
  if (qabs[2] > qabs[best]) best = 2;
  if (qabs[3] > qabs[best]) best = 3;
  float q[4];
  if (best == 0) {
    q[0] = qabs[0] * qabs[0]; q[1] = m[7] - m[5];
    q[2] = m[2] - m[6]; q[3] = m[3] - m[1];
  } else if (best == 1) {
    q[0] = m[7] - m[5]; q[1] = qabs[1] * qabs[1];
    q[2] = m[3] + m[1]; q[3] = m[2] + m[6];
  } else if (best == 2) {
    q[0] = m[2] - m[6]; q[1] = m[3] + m[1];
    q[2] = qabs[2] * qabs[2]; q[3] = m[5] + m[7];
  } else {
    q[0] = m[3] - m[1]; q[1] = m[6] + m[2];
    q[2] = m[7] + m[5]; q[3] = qabs[3] * qabs[3];
  }
  float denom = 2.0f * fmaxf(qabs[best], 0.1f);
  q[0] /= denom; q[1] /= denom; q[2] /= denom; q[3] /= denom;
  if (q[0] < 0.0f) {
    q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3];
  }
  float norm = sqrtf(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  float half = atan2f(norm, q[0]);
  float angle = 2.0f * half;
  float ratio = fabsf(angle) < 1.0e-6f
      ? 0.5f - angle * angle / 48.0f : sinf(half) / angle;
  vector[0] = q[1] / ratio;
  vector[1] = q[2] / ratio;
  vector[2] = q[3] / ratio;
}

__device__ inline void minimal_axis_angle_local(
    float sx, float sy, float sz, float tx, float ty, float tz,
    float eps, float* vector) {
  float si = safe_inv_norm(sx, sy, sz, eps);
  float ti = safe_inv_norm(tx, ty, tz, eps);
  if (si > 0.0f) { sx *= si; sy *= si; sz *= si; }
  else { sx = 1.0f; sy = 0.0f; sz = 0.0f; }
  if (ti > 0.0f) { tx *= ti; ty *= ti; tz *= ti; }
  else { tx = 1.0f; ty = 0.0f; tz = 0.0f; }
  float cx = sy * tz - sz * ty;
  float cy = sz * tx - sx * tz;
  float cz = sx * ty - sy * tx;
  float sine = sqrtf(cx * cx + cy * cy + cz * cz);
  float dot = fminf(1.0f, fmaxf(-1.0f, sx * tx + sy * ty + sz * tz));
  float ax = cx / fmaxf(sine, eps);
  float ay = cy / fmaxf(sine, eps);
  float az = cz / fmaxf(sine, eps);
  bool anti = sine <= eps && dot < 0.0f;
  if (anti) {
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    float asx = fabsf(sx), asy = fabsf(sy), asz = fabsf(sz);
    if (asx <= asy && asx <= asz) ux = 1.0f;
    else if (asy <= asz) uy = 1.0f;
    else uz = 1.0f;
    float projection = ux * sx + uy * sy + uz * sz;
    ax = ux - projection * sx;
    ay = uy - projection * sy;
    az = uz - projection * sz;
    float ai = safe_inv_norm(ax, ay, az, eps);
    if (ai > 0.0f) { ax *= ai; ay *= ai; az *= ai; }
    else { ax = 1.0f; ay = 0.0f; az = 0.0f; }
  }
  float angle = atan2f(sine, dot);
  if (sine <= eps && !anti) { ax = 0.0f; ay = 0.0f; az = 0.0f; }
  vector[0] = ax * angle; vector[1] = ay * angle; vector[2] = az * angle;
}

__device__ inline void kabsch_matrix_local(
    const float source[16][3], const float target[16][3],
    const float* confidence, int children, float eps, float* rotation) {
  float h[9] = {0};
  int positive = 0, strongest = 0;
  float strongest_w = -1.0f, total = 0.0f;
  for (int n = 0; n < children; ++n) {
    float w = confidence[n];
    w = isfinite(w) ? fmaxf(w, 0.0f) : 0.0f;
    if (w > eps) positive++;
    if (w > strongest_w) { strongest_w = w; strongest = n; }
    total += w;
    float sx=source[n][0], sy=source[n][1], sz=source[n][2];
    float tx=target[n][0], ty=target[n][1], tz=target[n][2];
    float si=safe_inv_norm(sx,sy,sz,eps), ti=safe_inv_norm(tx,ty,tz,eps);
    if(si>0){sx*=si;sy*=si;sz*=si;}else{sx=1;sy=0;sz=0;}
    if(ti>0){tx*=ti;ty*=ti;tz*=ti;}else{tx=1;ty=0;tz=0;}
    h[0]+=w*tx*sx;h[1]+=w*tx*sy;h[2]+=w*tx*sz;
    h[3]+=w*ty*sx;h[4]+=w*ty*sy;h[5]+=w*ty*sz;
    h[6]+=w*tz*sx;h[7]+=w*tz*sy;h[8]+=w*tz*sz;
  }
  float ata[9];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    float value=0;for(int k=0;k<3;++k)value+=h[k*3+i]*h[k*3+j];
    ata[i*3+j]=value;
  }
  float l0,l1;top_two_symmetric_eigenvalues(ata,&l0,&l1);
  float s0=sqrtf(l0),s1=sqrtf(l1);
  float norm=0;for(int i=0;i<9;++i)norm+=h[i]*h[i];
  norm=sqrtf(fmaxf(norm,0.0f));
  float scale=fmaxf(norm,eps);h[0]+=4*eps*scale;h[4]+=8*eps*scale;h[8]+=16*eps*scale;
  float b[9]={h[0],h[3],h[6],h[1],h[4],h[7],h[2],h[5],h[8]};
  float sigma=b[0]+b[4]+b[8],a[16]={0},v[16]={0};
  float zx=b[5]-b[7],zy=b[6]-b[2],zz=b[1]-b[3];
  a[0]=sigma;a[1]=a[4]=zx;a[2]=a[8]=zy;a[3]=a[12]=zz;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    a[(i+1)*4+j+1]=b[i*3+j]+b[j*3+i]-(i==j?sigma:0);
  for(int i=0;i<4;++i)v[i*4+i]=1;
  const int pairs[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
  for(int sweep=0;sweep<20;++sweep)for(int pp=0;pp<6;++pp){
    int p=pairs[pp][0],q=pairs[pp][1];float apq=a[p*4+q];
    if(fabsf(apq)<1e-12f)continue;
    float tau=(a[q*4+q]-a[p*4+p])/(2*apq);
    float t=copysignf(1.0f,tau)/(fabsf(tau)+sqrtf(1+tau*tau));
    float c=1/sqrtf(1+t*t),s=t*c;
    for(int k=0;k<4;++k){if(k==p||k==q)continue;float akp=a[k*4+p],akq=a[k*4+q];a[k*4+p]=a[p*4+k]=c*akp-s*akq;a[k*4+q]=a[q*4+k]=s*akp+c*akq;}
    float app=a[p*4+p],aqq=a[q*4+q];a[p*4+p]=c*c*app-2*s*c*apq+s*s*aqq;a[q*4+q]=s*s*app+2*s*c*apq+c*c*aqq;a[p*4+q]=a[q*4+p]=0;
    for(int k=0;k<4;++k){float vp=v[k*4+p],vq=v[k*4+q];v[k*4+p]=c*vp-s*vq;v[k*4+q]=s*vp+c*vq;}
  }
  int best=0;for(int i=1;i<4;++i)if(a[i*4+i]>a[best*4+best])best=i;
  float qr=v[best],qx=v[4+best],qy=v[8+best],qz=v[12+best];
  float qi=1/sqrtf(qr*qr+qx*qx+qy*qy+qz*qz);qr*=qi;qx*=qi;qy*=qi;qz*=qi;
  float kr[9];kr[0]=1-2*(qy*qy+qz*qz);kr[1]=2*(qx*qy-qz*qr);kr[2]=2*(qx*qz+qy*qr);kr[3]=2*(qx*qy+qz*qr);kr[4]=1-2*(qx*qx+qz*qz);kr[5]=2*(qy*qz-qx*qr);kr[6]=2*(qx*qz-qy*qr);kr[7]=2*(qy*qz+qx*qr);kr[8]=1-2*(qx*qx+qy*qy);
  bool good=positive>=2&&s0>eps&&s1>s0*1e-4f;
  float fallback[9];
  minimal_matrix(source[strongest][0],source[strongest][1],source[strongest][2],target[strongest][0],target[strongest][1],target[strongest][2],eps,fallback);
  for(int i=0;i<9;++i)rotation[i]=total>eps?(good?kr[i]:fallback[i]):((i%4==0)?1.0f:0.0f);
}

// One block owns one active (joint, frame) token.  The launch is ordered by
// skeleton depth, so the parent's corrected global matrix is already resident.
// Everything from parent recurrence through local/global writeback is fused.
__global__ void fixed_visual_depth_kernel(
    const float* __restrict__ baseline,
    const float* __restrict__ rest,
    const float* __restrict__ target,
    const bool* __restrict__ has_direction,
    const float* __restrict__ confidence,
    const float* __restrict__ aggregate_confidence,
    const float* __restrict__ static_hidden,
    const float* __restrict__ dynamic_weight,
    const float* __restrict__ hidden_weight,
    const float* __restrict__ hidden_bias,
    const float* __restrict__ output_weight,
    const float* __restrict__ output_bias,
    const int64_t* __restrict__ depth_flat,
    const int64_t* __restrict__ parent_flat,
    const int64_t* __restrict__ child_count,
    const bool* __restrict__ root_mask,
    const bool* __restrict__ valid_parent,
    const bool* __restrict__ eligible,
    const bool* __restrict__ frame_mask,
    float* __restrict__ corrected_global,
    float* __restrict__ corrected_local,
    float* __restrict__ gate_out,
    bool* __restrict__ observable_out,
    int depth_count, int batch_size, int frames, int joints,
    int children, int hidden, float fixed_gate_scale,
    int gate_override, float eps) {
  int token = blockIdx.x;
  int row = token / frames;
  int frame = token - row * frames;
  if (row >= depth_count) return;
  int lane = threadIdx.x;
  int flat = static_cast<int>(depth_flat[row]);
  int batch = flat / joints;
  int joint = flat - batch * joints;
  int parent = static_cast<int>(parent_flat[row]);
  int matrix_index = (flat * frames + frame) * 9;
  int parent_matrix_index = (parent * frames + frame) * 9;
  int feature_index = (batch * frames + frame) * joints + joint;

  __shared__ float provisional[9];
  __shared__ float source[16][3];
  __shared__ float target_local[16][3];
  __shared__ float confidence_local[16];
  __shared__ float proposal_axis[3];
  __shared__ float disagreement;
  __shared__ float first_hidden[256];
  __shared__ float second_hidden[256];

  if (lane == 0) {
    const float* base = baseline + matrix_index;
    bool root = root_mask[flat];
    bool owner = root || valid_parent[flat];
    if (root) {
      for (int i=0;i<9;++i) provisional[i]=base[i];
    } else {
      matmul3(corrected_global + parent_matrix_index, base, provisional);
    }
    if (!owner) for (int i=0;i<9;++i) provisional[i]=base[i];

    float weighted_angle=0.0f, weight_sum=0.0f;
    int count = static_cast<int>(child_count[flat]);
    if (count > children) count = children;
    for (int k=0;k<children;++k) {
      const float* r = rest + (flat * children + k) * 3;
      float sx=provisional[0]*r[0]+provisional[1]*r[1]+provisional[2]*r[2];
      float sy=provisional[3]*r[0]+provisional[4]*r[1]+provisional[5]*r[2];
      float sz=provisional[6]*r[0]+provisional[7]*r[1]+provisional[8]*r[2];
      float inv=safe_inv_norm(sx,sy,sz,eps);
      if(inv>0){sx*=inv;sy*=inv;sz*=inv;}else{sx=1;sy=0;sz=0;}
      int direction_index=((batch*frames+frame)*joints+joint)*children+k;
      const float* tp=target+direction_index*3;
      float tx=tp[0],ty=tp[1],tz=tp[2];
      if(!has_direction[direction_index]){tx=sx;ty=sy;tz=sz;}
      source[k][0]=sx;source[k][1]=sy;source[k][2]=sz;
      target_local[k][0]=tx;target_local[k][1]=ty;target_local[k][2]=tz;
      float w=confidence[direction_index];w=isfinite(w)?fmaxf(w,0.0f):0.0f;
      confidence_local[k]=w;
      float dot=fminf(1.0f,fmaxf(-1.0f,sx*tx+sy*ty+sz*tz));
      float cx=sy*tz-sz*ty,cy=sz*tx-sx*tz,cz=sx*ty-sy*tx;
      weighted_angle+=atan2f(sqrtf(cx*cx+cy*cy+cz*cz),dot)*w;
      weight_sum+=w;
    }
    disagreement=weighted_angle/fmaxf(weight_sum,eps);
    proposal_axis[0]=proposal_axis[1]=proposal_axis[2]=0.0f;
    if (eligible[flat] && count == 1) {
      minimal_axis_angle_local(source[0][0],source[0][1],source[0][2],target_local[0][0],target_local[0][1],target_local[0][2],eps,proposal_axis);
    } else if (eligible[flat] && count > 1) {
      float correction[9];
      kabsch_matrix_local(source,target_local,confidence_local,children,eps,correction);
      matrix_axis_angle(correction,proposal_axis);
    }
  }
  __syncthreads();

  bool observable=frame_mask[batch*frames+frame]&&eligible[flat];
  float aggregate=aggregate_confidence[feature_index];
  if(lane<hidden){
    float x=static_hidden[feature_index*hidden+lane]
      +dynamic_weight[lane*2]*aggregate
      +dynamic_weight[lane*2+1]*(disagreement/3.14159265358979323846f);
    first_hidden[lane]=silu(x);
  }
  __syncthreads();
  if(lane<hidden){
    float x=hidden_bias[lane];const float* w=hidden_weight+lane*hidden;
    for(int i=0;i<hidden;++i)x+=w[i]*first_hidden[i];
    second_hidden[lane]=silu(x);
  }
  __syncthreads();
  if(lane==0){
    float logit=output_bias[0];for(int i=0;i<hidden;++i)logit+=output_weight[i]*second_hidden[i];
    float learned=1.0f/(1.0f+expf(-logit));
    float confidence_gate=aggregate*fixed_gate_scale;
    float gate=confidence_gate*learned;
    if(gate_override==1)gate=confidence_gate;
    else if(gate_override==2)gate=1.0f;
    else if(gate_override==3)gate=0.0f;
    bool active=observable&&gate>0.0f;
    float current_gate=active?gate:0.0f;
    float applied[3]={proposal_axis[0]*current_gate,proposal_axis[1]*current_gate,proposal_axis[2]*current_gate};
    float blended[9];rodrigues_vector(applied,blended);
    float corrected[9];matmul3(blended,provisional,corrected);
    if(!active)for(int i=0;i<9;++i)corrected[i]=provisional[i];
    for(int i=0;i<9;++i)corrected_global[matrix_index+i]=corrected[i];
    float local[9];
    bool root=root_mask[flat],owner=root||valid_parent[flat];
    if(root)for(int i=0;i<9;++i)local[i]=corrected[i];
    else parent_transpose_matmul3(corrected_global+parent_matrix_index,corrected,local);
    const float* base=baseline+matrix_index;
    for(int i=0;i<9;++i)corrected_local[matrix_index+i]=(active&&owner)?local[i]:base[i];
    gate_out[flat*frames+frame]=current_gate;
    observable_out[flat*frames+frame]=observable;
  }
}

__global__ void visual_gate_kernel(
    const float* __restrict__ static_hidden,
    const float* __restrict__ dynamic,
    const float* __restrict__ dynamic_weight,
    const float* __restrict__ hidden_weight,
    const float* __restrict__ hidden_bias,
    const float* __restrict__ output_weight,
    const float* __restrict__ output_bias,
    const bool* __restrict__ token_mask,
    float* __restrict__ output,
    int token_count, int hidden) {
  int token=blockIdx.x, lane=threadIdx.x;
  if(token>=token_count)return;
  __shared__ float first[256];__shared__ float second[256];
  if(lane<hidden){
    float x=static_hidden[token*hidden+lane]
      +dynamic_weight[lane*2]*dynamic[token*2]
      +dynamic_weight[lane*2+1]*dynamic[token*2+1];
    first[lane]=silu(x);
  }
  __syncthreads();
  if(lane<hidden){
    float x=hidden_bias[lane];
    const float* w=hidden_weight+lane*hidden;
    for(int i=0;i<hidden;++i)x+=w[i]*first[i];
    second[lane]=silu(x);
  }
  __syncthreads();
  if(lane==0){
    float x=output_bias[0];for(int i=0;i<hidden;++i)x+=output_weight[i]*second[i];
    output[token]=token_mask[token]?x:0.0f;
  }
}

}  // namespace

std::vector<torch::Tensor> fused_depth_geometry_cuda(
    torch::Tensor provisional,
    torch::Tensor rest_direction,
    torch::Tensor target,
    torch::Tensor target_has_direction,
    torch::Tensor confidence,
    double eps) {
  const auto n = provisional.size(0);
  const auto t_count = provisional.size(1);
  const auto k_count = rest_direction.size(1);
  auto source = torch::empty({n, t_count, k_count, 3}, provisional.options());
  auto target_out = torch::empty_like(source);
  auto disagreement = torch::empty({n, t_count}, provisional.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_kernel<<<n * t_count, 32, 0, stream>>>(
      provisional.data_ptr<float>(), rest_direction.data_ptr<float>(),
      target.data_ptr<float>(), target_has_direction.data_ptr<bool>(),
      confidence.data_ptr<float>(), source.data_ptr<float>(),
      target_out.data_ptr<float>(), disagreement.data_ptr<float>(),
      n, t_count, k_count, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {source, target_out, disagreement};
}

torch::Tensor axis_angle_to_matrix_cuda(torch::Tensor axis_angle) {
  const int64_t count = axis_angle.numel() / 3;
  std::vector<int64_t> shape(axis_angle.sizes().begin(), axis_angle.sizes().end());
  shape.back() = 3;
  shape.push_back(3);
  auto matrix = torch::empty(shape, axis_angle.options());
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  axis_angle_matrix_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
      axis_angle.data_ptr<float>(), matrix.data_ptr<float>(), count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return matrix;
}

torch::Tensor matrix_to_axis_angle_cuda(torch::Tensor matrix) {
  const int64_t count = matrix.numel() / 9;
  std::vector<int64_t> shape(matrix.sizes().begin(), matrix.sizes().end() - 2);
  shape.push_back(3);
  auto vector = torch::empty(shape, matrix.options());
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  matrix_axis_angle_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
      matrix.data_ptr<float>(), vector.data_ptr<float>(), count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return vector;
}

std::vector<torch::Tensor> confidence_weighted_kabsch_cuda(
    torch::Tensor source, torch::Tensor target,
    torch::Tensor confidence, double eps) {
  const int64_t count=confidence.numel()/confidence.size(-1);
  const int children=confidence.size(-1);
  std::vector<int64_t> rshape(source.sizes().begin(),source.sizes().end()-2);rshape.push_back(3);rshape.push_back(3);
  std::vector<int64_t> vshape(confidence.sizes().begin(),confidence.sizes().end()-1);
  auto rotation=torch::empty(rshape,source.options());auto valid=torch::empty(vshape,confidence.options().dtype(torch::kBool));
  constexpr int threads=128;auto stream=at::cuda::getCurrentCUDAStream();
  kabsch_kernel<<<(count+threads-1)/threads,threads,0,stream>>>(source.data_ptr<float>(),target.data_ptr<float>(),confidence.data_ptr<float>(),rotation.data_ptr<float>(),valid.data_ptr<bool>(),count,children,static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {rotation,valid};
}

torch::Tensor visual_gate_cuda(
    torch::Tensor static_hidden,torch::Tensor dynamic,
    torch::Tensor dynamic_weight,torch::Tensor hidden_weight,
    torch::Tensor hidden_bias,torch::Tensor output_weight,
    torch::Tensor output_bias,torch::Tensor token_mask) {
  int64_t tokens=static_hidden.numel()/static_hidden.size(-1);int hidden=static_hidden.size(-1);
  std::vector<int64_t> shape(static_hidden.sizes().begin(),static_hidden.sizes().end()-1);shape.push_back(1);
  auto output=torch::empty(shape,static_hidden.options());auto stream=at::cuda::getCurrentCUDAStream();
  visual_gate_kernel<<<tokens,256,0,stream>>>(static_hidden.data_ptr<float>(),dynamic.data_ptr<float>(),dynamic_weight.data_ptr<float>(),hidden_weight.data_ptr<float>(),hidden_bias.data_ptr<float>(),output_weight.data_ptr<float>(),output_bias.data_ptr<float>(),token_mask.data_ptr<bool>(),output.data_ptr<float>(),tokens,hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

void fixed_visual_depth_cuda(
    torch::Tensor baseline, torch::Tensor rest, torch::Tensor target,
    torch::Tensor has_direction, torch::Tensor confidence,
    torch::Tensor aggregate_confidence, torch::Tensor static_hidden,
    torch::Tensor dynamic_weight, torch::Tensor hidden_weight,
    torch::Tensor hidden_bias, torch::Tensor output_weight,
    torch::Tensor output_bias, torch::Tensor depth_flat,
    torch::Tensor parent_flat, torch::Tensor child_count,
    torch::Tensor root_mask, torch::Tensor valid_parent,
    torch::Tensor eligible, torch::Tensor frame_mask,
    torch::Tensor corrected_global, torch::Tensor corrected_local,
    torch::Tensor gate_out, torch::Tensor observable_out,
    double fixed_gate_scale, int64_t gate_override, double eps) {
  int batch_size=rest.size(0),joints=rest.size(1),children=rest.size(2);
  int frames=target.size(1),hidden=static_hidden.size(-1),depth_count=depth_flat.numel();
  if(depth_count==0)return;
  auto stream=at::cuda::getCurrentCUDAStream();
  fixed_visual_depth_kernel<<<depth_count*frames,256,0,stream>>>(
      baseline.data_ptr<float>(),rest.data_ptr<float>(),target.data_ptr<float>(),
      has_direction.data_ptr<bool>(),confidence.data_ptr<float>(),
      aggregate_confidence.data_ptr<float>(),static_hidden.data_ptr<float>(),
      dynamic_weight.data_ptr<float>(),hidden_weight.data_ptr<float>(),
      hidden_bias.data_ptr<float>(),output_weight.data_ptr<float>(),
      output_bias.data_ptr<float>(),depth_flat.data_ptr<int64_t>(),
      parent_flat.data_ptr<int64_t>(),child_count.data_ptr<int64_t>(),
      root_mask.data_ptr<bool>(),valid_parent.data_ptr<bool>(),eligible.data_ptr<bool>(),
      frame_mask.data_ptr<bool>(),corrected_global.data_ptr<float>(),
      corrected_local.data_ptr<float>(),gate_out.data_ptr<float>(),
      observable_out.data_ptr<bool>(),depth_count,batch_size,frames,joints,
      children,hidden,static_cast<float>(fixed_gate_scale),
      static_cast<int>(gate_override),static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor minimal_axis_angle_cuda(
    torch::Tensor source, torch::Tensor target, double eps) {
  const int64_t count = source.numel() / 3;
  auto vector = torch::empty_like(source);
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  minimal_axis_angle_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
      source.data_ptr<float>(), target.data_ptr<float>(), vector.data_ptr<float>(),
      count, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return vector;
}
