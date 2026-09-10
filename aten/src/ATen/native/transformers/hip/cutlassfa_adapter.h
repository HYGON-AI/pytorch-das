// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#ifndef CUTLASSFA_ADAPTER_H
#define CUTLASSFA_ADAPTER_H

#ifdef USE_ROCM
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <fnmatch.h>
#include <vector>
#include <typeinfo>
#include <ATen/hip/HIPContext.h>
#include <c10/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/CallOnce.h>


namespace fs = std::filesystem;

typedef std::vector<at::Tensor> (*mha_fwd_func)(at::Tensor , const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                                const float, const float, bool,
                                                int, int, const float, const bool,
                                                c10::optional<at::Generator>, bool);

typedef std::vector<at::Tensor> (*mha_fwd_func_new)(at::Tensor , const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                                const float, const float, bool,
                                                int, int, const float, const bool,
                                                c10::optional<at::Generator>, bool,
                                                const c10::optional<at::Tensor>,
                                                float);

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

typedef std::vector<at::Tensor> (*mha_varlen_fwd_func_new)(at::Tensor , const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>, const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>,c10::optional<const at::Tensor>,
                                                c10::optional<at::Tensor>,c10::optional<at::Tensor>,
                                                int, const int, const float, const float, bool, bool,
                                                int, int, const float, const bool,
						const c10::optional<at::Tensor>,
						const c10::optional<at::Tensor>,
						const c10::optional<at::Tensor>,
                                                c10::optional<at::Generator>,
                                                const c10::optional<at::Tensor>);

typedef std::vector<at::Tensor> (*mha_varlen_bwd_func)(const at::Tensor , const at::Tensor, const at::Tensor,
                                                const at::Tensor, const at::Tensor, const at::Tensor,
                                                c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                                                c10::optional<at::Tensor>,const at::Tensor , const at::Tensor,c10::optional<at::Tensor>,
                                                const int, const int, const float, const float, const bool, const bool,
                                                int, int, const float, const bool,
                                                c10::optional<at::Generator>, c10::optional<at::Tensor>);

typedef int (*get_fa_api_changed)();


using mha_fwd_attnmask_func = std::vector<at::Tensor> (*)(
    at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    std::optional<at::Tensor>&, std::optional<at::Tensor>&,
    float, float, bool, int, int, float, float, bool,
    std::optional<at::Generator>, bool, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&);

using mha_bwd_attnmask_func = std::vector<at::Tensor> (*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    std::optional<at::Tensor>&, std::optional<at::Tensor>&,
    std::optional<at::Tensor>&, std::optional<at::Tensor>&,
    float, float, bool, int, int, float, bool,
    std::optional<at::Generator>, std::optional<at::Tensor>&, bool);

using mha_varlen_fwd_attnmask_func = std::vector<at::Tensor> (*)(
    at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    std::optional<at::Tensor>&, const at::Tensor&, const at::Tensor&,
    std::optional<at::Tensor>&, std::optional<at::Tensor>&,
    int, int, float, float, bool, bool, int, int, float, float, bool,
    std::optional<at::Generator>, const std::optional<at::Tensor>&);

using mha_varlen_bwd_attnmask_func = std::vector<at::Tensor> (*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    std::optional<at::Tensor>&, std::optional<at::Tensor>&,
    std::optional<at::Tensor>&, const at::Tensor&, const at::Tensor&,
    std::optional<at::Tensor>&, int, int, float, float, bool, bool,
    int, int, float, bool, std::optional<at::Generator>,
    std::optional<at::Tensor>&);

// 查找匹配的动态库文件
std::string find_matching_libraries(const fs::path& directory, const std::string& pattern) {
    bool found = false;
    std::string file_path = "";
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) { // 只处理普通文件
            std::string file_name = entry.path().filename().string();
            if (fnmatch(pattern.c_str(), file_name.c_str(), 0) == 0) { // 匹配通配符
                file_path = entry.path().string();
                found = true;
                break;
            }
        }
    }

    if (!found) {
        TORCH_CHECK(false, "No matching libraries found for flash_attn_2_cuda*.so");
    }
    return file_path;
}

std::string print_library_path() {
    Dl_info info;
    std::string path;
    if (dladdr(reinterpret_cast<void*>(&print_library_path), &info)) {
        path = std::string(info.dli_fname);
    } else {
	TORCH_CHECK(false, "Error: Unable to retrieve library path.");
    }
    return path;
}

std::string get_so_path() {
  std::string fa_so_p;
  auto* e_fa_so = std::getenv("FA_SO_PATH");

  if (e_fa_so) {
    fa_so_p = std::string(e_fa_so);
  } else {
    auto c_so_path = print_library_path();
    fs::path c_so_fs(c_so_path);
    auto lib_fs = c_so_fs.parent_path().parent_path().parent_path();
    fa_so_p = find_matching_libraries(lib_fs, "flash_attn_2_cuda*.so");
  }
  return fa_so_p;
}

namespace sdp {

namespace cutlassfa_adapter {

class CUTLASSFA {
public:
    mha_fwd_func_new mha_fwd_new = nullptr;
    mha_fwd_func mha_fwd = nullptr;
    mha_bwd_func mha_bwd = nullptr;
    mha_varlen_fwd_func_new mha_varlen_fwd_new = nullptr;
    mha_varlen_fwd_func mha_varlen_fwd = nullptr;

    mha_varlen_bwd_func mha_varlen_bwd = nullptr;
    mha_fwd_attnmask_func mha_fwd_attnmask = nullptr;
    mha_bwd_attnmask_func mha_bwd_attnmask = nullptr;
    mha_varlen_fwd_attnmask_func mha_varlen_fwd_attnmask = nullptr;
    mha_varlen_bwd_attnmask_func mha_varlen_bwd_attnmask = nullptr;
    // mha_fwd_kvcache_func mha_fwd_kvcache = nullptr;
    CUTLASSFA() {

      auto fa_so_p = get_so_path();
      handle = dlopen(fa_so_p.c_str(), RTLD_LAZY);
      if(!handle){
        TORCH_CHECK(false, "Not load library for Cutlass Fa so");
      }

      get_fa_api_changed is_fa_api_changed = (get_fa_api_changed)dlsym(handle, "flash_attn_api_changed");
      if(is_fa_api_changed != nullptr && is_fa_api_changed()) {
        is_new = true;
        TORCH_WARN_ONCE(
            "sdpa adopt the new interface of flash-attn"
        );
        mha_fwd_new = (mha_fwd_func_new)dlsym(handle, "mha_fwd");
        mha_varlen_fwd_new = (mha_varlen_fwd_func_new)dlsym(handle, "mha_varlen_fwd");
      } else {
        TORCH_WARN_ONCE(
            "sdpa adopt the old interface of flash-attn"
        );
        mha_fwd = (mha_fwd_func)dlsym(handle, "mha_fwd");
        mha_varlen_fwd = (mha_varlen_fwd_func)dlsym(handle, "mha_varlen_fwd");
      }
      mha_bwd = (mha_bwd_func)dlsym(handle, "mha_bwd");
      mha_varlen_bwd = (mha_varlen_bwd_func)dlsym(handle, "mha_varlen_bwd");

      const char* dlsym_error = dlerror();
      if (dlsym_error) {
          dlclose(handle);
          TORCH_CHECK(false, "无法获取flash-attn 函数符号: ");
      }
      mha_fwd_attnmask =
          load_optional_mask_api<mha_fwd_attnmask_func>("mha_fwd_attnmask");
      mha_bwd_attnmask =
          load_optional_mask_api<mha_bwd_attnmask_func>("mha_bwd_attnmask");
      mha_varlen_fwd_attnmask =
          load_optional_mask_api<mha_varlen_fwd_attnmask_func>(
              "mha_varlen_fwd_attnmask");
      mha_varlen_bwd_attnmask =
          load_optional_mask_api<mha_varlen_bwd_attnmask_func>(
              "mha_varlen_bwd_attnmask");
    }

    bool has_attnmask_api() const {
      return mha_fwd_attnmask && mha_bwd_attnmask;
    }

    bool has_varlen_attnmask_api() const {
      return mha_varlen_fwd_attnmask && mha_varlen_bwd_attnmask;
    }

    static CUTLASSFA &get_instance(){
      static CUTLASSFA instance;
      return instance;
    }
    static bool get_fa_changed() {
      return is_new;
    }
    // template<class T, class... Args>
    CUTLASSFA(const CUTLASSFA&) = delete;
    CUTLASSFA(CUTLASSFA&&) = delete;
    CUTLASSFA& operator=(const CUTLASSFA&) = delete;
    CUTLASSFA& operator=(CUTLASSFA&&) = delete;
private:
    template <typename Fn>
    Fn load_optional_mask_api(const char* name) {
      // HYGON wheels expose pybind function-pointer RTTI. Fail closed when
      // the signature is unknown (the mask forward ABI gained an argument).
      const auto signature = std::string("_ZTS") + typeid(Fn).name();
      dlerror();
      auto* signature_symbol = dlsym(handle, signature.c_str());
      const auto* signature_error = dlerror();
      if (!signature_symbol || signature_error) {
        return nullptr;
      }
      dlerror();
      auto* symbol = dlsym(handle, name);
      const auto* error = dlerror();
      Dl_info signature_info{};
      Dl_info function_info{};
      if (error || !symbol || !dladdr(signature_symbol, &signature_info) ||
          !dladdr(symbol, &function_info) ||
          signature_info.dli_fbase != function_info.dli_fbase) {
        return nullptr;
      }
      return reinterpret_cast<Fn>(symbol);
    }

    void* handle;
    static bool is_new;
    ~CUTLASSFA(){
        dlclose(handle);
    }
};

bool CUTLASSFA::is_new = false;

}
}

#endif
#endif // USE_ROCM
