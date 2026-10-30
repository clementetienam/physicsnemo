// SPDX-License-Identifier: Apache-2.0
// Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
//          Nick Luiken, Issam Said
//
// pvt_upload.cu -- moving the PVT tables to the device.
//
// This is split out from pvt_tables.cpp for one reason: that file must build
// with a plain host compiler so tests/test_pvt.cpp can run the parser and the
// evaluators anywhere, and only this part needs CUDA.  Guarding the device
// half with __CUDACC__ inside the .cpp would have done the same job only if
// nvcc defined that macro when handed a .cpp, which is not something to rely
// on -- the file extension decides it, so the split is by file.
//
// Everything numerical lives in flatten_pvt(), which is host code in
// pvt_tables.cpp and shared with HostPvt::view().  What remains here is the
// copy and the free.
#include "common.hpp"
#include "pvt_tables.hpp"

#include <algorithm>
#include <vector>

namespace nvrs {

// One thread, one evaluation, straight out of the kernel's own code path.
__global__ static void k_probe_pvt(DevicePvt d, Real p, Real rs_max,
                                   PvtCell* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *out = dev_pvt_cell(d, p, rs_max);
}

PvtCell probe_device_pvt(const DevicePvt& d, Real p, Real rs_max) {
  PvtCell host;                      // defaults: 1/1/1, zero slopes
  if (!d.valid) return host;
  PvtCell* dev = nullptr;
  NVRS_CHECK_CUDA(cudaMalloc(&dev, sizeof(PvtCell)));
  k_probe_pvt<<<1, 1>>>(d, p, rs_max, dev);
  NVRS_CHECK_CUDA(cudaGetLastError());
  NVRS_CHECK_CUDA(cudaMemcpy(&host, dev, sizeof(PvtCell),
                             cudaMemcpyDeviceToHost));
  cudaFree(dev);
  return host;
}

DevicePvt* upload_pvt_regions(const PvtTables& t, int* n_regions_out) {
  const int n = std::max(1, t.n_regions());
  if (n_regions_out) *n_regions_out = n;
  std::vector<DevicePvt> host(n);
  for (int r = 0; r < n; ++r) host[r] = upload_pvt(t, r);
  DevicePvt* dev = nullptr;
  NVRS_CHECK_CUDA(cudaMalloc(&dev, sizeof(DevicePvt) * n));
  NVRS_CHECK_CUDA(cudaMemcpy(dev, host.data(), sizeof(DevicePvt) * n,
                             cudaMemcpyHostToDevice));
  return dev;
}

DevicePvt upload_pvt(const PvtTables& t, int region) {
  DevicePvt d;
  const HostPvt h = flatten_pvt(t, region);
  // VAPOIL must reach the device: every kernel reads Rv through
  // dev_rv_sat, which scales by this.  Set before the validity bail-out
  // so an unloaded table still reports vaporised oil as switched off.
  d.rv_scale = h.rv_scale;
  if (!h.valid) return d;

  auto up = [](const std::vector<Real>& v) {
    Real* p = nullptr;
    NVRS_CHECK_CUDA(cudaMalloc(&p, sizeof(Real) * std::max<size_t>(v.size(), 1)));
    if (!v.empty())
      NVRS_CHECK_CUDA(cudaMemcpy(p, v.data(), sizeof(Real) * v.size(),
                                 cudaMemcpyHostToDevice));
    return p;
  };
  auto upi = [](const std::vector<int>& v) {
    int* p = nullptr;
    NVRS_CHECK_CUDA(cudaMalloc(&p, sizeof(int) * std::max<size_t>(v.size(), 1)));
    if (!v.empty())
      NVRS_CHECK_CUDA(cudaMemcpy(p, v.data(), sizeof(int) * v.size(),
                                 cudaMemcpyHostToDevice));
    return p;
  };

  if (!h.sat_rs.empty()) {
    d.sat_rs = up(h.sat_rs); d.sat_p = up(h.sat_p);
    d.sat_bo = up(h.sat_bo); d.sat_mu = up(h.sat_mu);
    d.n_sat = (int)h.sat_rs.size();
    d.us_off = upi(h.us_off);
    d.us_p = up(h.us_p); d.us_bo = up(h.us_bo); d.us_mu = up(h.us_mu);
  }
  if (!h.gsat_p.empty()) {
    d.gsat_p = up(h.gsat_p); d.gsat_rv = up(h.gsat_rv);
    d.gsat_bg = up(h.gsat_bg); d.gsat_mu = up(h.gsat_mu);
    d.n_gsat = (int)h.gsat_p.size();
    d.gus_off = upi(h.gus_off); d.gus_rv = up(h.gus_rv);
    d.gus_invbg = up(h.gus_invbg); d.gus_invbgmu = up(h.gus_invbgmu);
  }
  d.w_pref = h.w_pref; d.w_bw = h.w_bw; d.w_cw = h.w_cw;
  d.w_mu = h.w_mu; d.w_cmu = h.w_cmu;
  d.valid = true;
  return d;
}

void free_device_pvt(DevicePvt& d) {
  for (const void* p : {(const void*)d.sat_rs, (const void*)d.sat_p,
                        (const void*)d.sat_bo, (const void*)d.sat_mu,
                        (const void*)d.us_p, (const void*)d.us_bo,
                        (const void*)d.us_mu, (const void*)d.gsat_p,
                        (const void*)d.gsat_rv, (const void*)d.gsat_bg,
                        (const void*)d.gsat_mu, (const void*)d.gus_rv,
                        (const void*)d.gus_invbg,
                        (const void*)d.gus_invbgmu})
    if (p) cudaFree(const_cast<void*>(p));
  if (d.us_off) cudaFree(const_cast<int*>(d.us_off));
  if (d.gus_off) cudaFree(const_cast<int*>(d.gus_off));
  d = DevicePvt{};
}

}  // namespace nvrs
