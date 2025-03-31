#pragma once
#ifndef CUTLASSFA_ADAPTER_H
#define CUTLASSFA_ADAPTER_H

#ifdef USE_ROCM
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <ATen/hip/HIPContext.h>
#include <c10/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/CallOnce.h>

typedef std::vector<at::Tensor> (*mha_fwd_func)(at::Tensor , const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                                const float, const float, bool,
                                                int, int, const float, const bool,
                                                c10::optional<at::Generator>, bool);


typedef std::vector<at::Tensor> (*mha_bwd_func)(const at::Tensor , const at::Tensor, const at::Tensor,
                                                const at::Tensor , const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                                c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                                const float, const float, const bool,
                                                int, int, const float, const bool,
                                                c10::optional<at::Generator>, c10::optional<at::Tensor>, bool);


typedef std::vector<at::Tensor> (*mha_varlen_fwd_func)(at::Tensor , const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>, const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>,c10::optional<const at::Tensor>,
                                                c10::optional<at::Tensor>,c10::optional<at::Tensor>,
                                                int, const int, const float, const float, bool, bool,
                                                int, int, const float, const bool,
                                                c10::optional<at::Generator>);

typedef std::vector<at::Tensor> (*mha_varlen_bwd_func)(const at::Tensor , const at::Tensor, const at::Tensor,
                                                const at::Tensor, const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                                c10::optional<at::Tensor>,const at::Tensor , const at::Tensor,c10::optional<at::Tensor>,
                                                const int, const int, const float, const float, const bool, const bool,
                                                int, int, const float, const bool,
                                                c10::optional<at::Generator>, c10::optional<at::Tensor>);

// typedef std::vector<at::Tensor> (*mha_fwd_kvcache_func)(at::Tensor , const at::Tensor, const at::Tensor,
//                                                 c10::optional<at::Tensor>, c10::optional<at::Tensor>,c10::optional<at::Tensor>, c10::optional<at::Tensor>,
//                                                 c10::optional<at::Tensor>, c10::optional<at::Tensor>,c10::optional<at::Tensor>, c10::optional<at::Tensor>,
//                                                 c10::optional<at::Tensor>, c10::optional<at::Tensor>,
//                                                 const float, bool, int, int, const float, bool, int);

namespace sdp {

namespace cutlassfa_adapter {

class CUTLASSFA {
public:
    mha_fwd_func mha_fwd = nullptr;
    mha_bwd_func mha_bwd = nullptr;
    mha_varlen_fwd_func mha_varlen_fwd = nullptr;
    mha_varlen_bwd_func mha_varlen_bwd = nullptr;
    // mha_fwd_kvcache_func mha_fwd_kvcache = nullptr;
    CUTLASSFA() {
      handle = dlopen("flash_attn_2_cuda.so", RTLD_LAZY);
      if(!handle){
        TORCH_CHECK(false, "Not load library for Cutlass Fa so");
      }
      mha_fwd = (mha_fwd_func)dlsym(handle, "mha_fwd");
      mha_bwd = (mha_bwd_func)dlsym(handle, "mha_bwd");
      mha_varlen_fwd = (mha_varlen_fwd_func)dlsym(handle, "mha_varlen_fwd");
      mha_varlen_bwd = (mha_varlen_bwd_func)dlsym(handle, "mha_varlen_bwd");
      // mha_fwd_kvcache = (mha_fwd_kvcache_func)dlsym(handle, "mha_fwd_kvcache");

      const char* dlsym_error = dlerror();
      if (dlsym_error) {
          dlclose(handle);
          TORCH_CHECK(false, "无法获取flash-attn 函数符号: ");
      }
    }

    static CUTLASSFA &get_instance(){
      static CUTLASSFA instance;
      return instance;
    }
    // template<class T, class... Args>
    CUTLASSFA(const CUTLASSFA&) = delete;
    CUTLASSFA(CUTLASSFA&&) = delete;
    CUTLASSFA& operator=(const CUTLASSFA&) = delete;
    CUTLASSFA& operator=(CUTLASSFA&&) = delete;
private:
    void* handle;
    
    ~CUTLASSFA(){
        dlclose(handle);
    }
};

}
}

#endif
#endif // USE_ROCM
