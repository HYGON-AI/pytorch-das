namespace {
template <typename scalar_t>
hml_sparse_status_t hml_create_csr(
    hml_sparse_matrix_t* m, int64_t rows, int64_t cols,
    HML_INT* crow, HML_INT* col, scalar_t* val) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return hml_sparse_s_create_csr(m, HML_SPARSE_INDEX_BASE_ZERO, rows, cols, crow, crow + 1, col, val);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    return hml_sparse_d_create_csr(m, HML_SPARSE_INDEX_BASE_ZERO, rows, cols, crow, crow + 1, col, val);
  } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
    return hml_sparse_c_create_csr(m, HML_SPARSE_INDEX_BASE_ZERO, rows, cols, crow, crow + 1, col,
        reinterpret_cast<HML_Complex8*>(val));
  } else {
    return hml_sparse_z_create_csr(m, HML_SPARSE_INDEX_BASE_ZERO, rows, cols, crow, crow + 1, col,
        reinterpret_cast<HML_Complex16*>(val));
  }
}
// HSPARSE 等效工具（MKL 版在 #if AT_USE_MKL_SPARSE() 内，此处提供 HML 版）
void inline hml_col_indices_and_values_resize_(const Tensor& input, int64_t nnz) {
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
      ->set_member_tensors(
          input.crow_indices(),
          input.col_indices().resize_({nnz}),
          input.values().resize_({nnz}),
          input.sizes());
}

// HML_INT = int32_t（LP64），indices 必须为 int32
void inline hml_indices_to_compatible_inplace(const Tensor& input) {
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
      ->set_member_tensors(
          input.crow_indices().to(kInt),
          input.col_indices().to(kInt),
          input.values(),
          input.sizes());
}

template <typename scalar_t>
hml_sparse_status_t hml_add(
    hml_sparse_operation_t op, scalar_t alpha,
    hml_sparse_matrix_t A, hml_sparse_matrix_t B, hml_sparse_matrix_t* C) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return hml_sparse_s_add(op, A, alpha, B, C);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    return hml_sparse_d_add(op, A, alpha, B, C);
  } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
    HML_Complex8 ha{alpha.real(), alpha.imag()};
    return hml_sparse_c_add(op, A, ha, B, C);
  } else {
    HML_Complex16 ha{alpha.real(), alpha.imag()};
    return hml_sparse_z_add(op, A, ha, B, C);
  }
}

template <typename scalar_t>
void hml_result_copy_(const Tensor& input, hml_sparse_matrix_t hml_desc) {
  hml_sparse_index_base_t indexing;
  HML_INT rows = 0, cols = 0;
  HML_INT *rows_start = nullptr, *rows_end = nullptr, *columns = nullptr;
  scalar_t* values = nullptr;
  hml_sparse_status_t st;
  if constexpr (std::is_same_v<scalar_t, float>) {
    st = hml_sparse_s_export_csr(hml_desc, &indexing, &rows, &cols,
                                 &rows_start, &rows_end, &columns, &values);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    st = hml_sparse_d_export_csr(hml_desc, &indexing, &rows, &cols,
                                 &rows_start, &rows_end, &columns, &values);
  } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
    st = hml_sparse_c_export_csr(hml_desc, &indexing, &rows, &cols,
                                 &rows_start, &rows_end, &columns,
                                 reinterpret_cast<HML_Complex8**>(&values));
  } else {
    st = hml_sparse_z_export_csr(hml_desc, &indexing, &rows, &cols,
                                 &rows_start, &rows_end, &columns,
                                 reinterpret_cast<HML_Complex16**>(&values));
  }
  TORCH_CHECK(st == 0, "hml_sparse export_csr failed: ", st);
  HML_INT nnz = rows_end[rows - 1];
  hml_col_indices_and_values_resize_(input, nnz);
  auto crow_indices = input.crow_indices();
  auto col_indices = input.col_indices();
  auto input_values = input.values();
  if (nnz > 0) {
    std::memcpy(
        input_values.mutable_data_ptr<scalar_t>(), values, nnz * sizeof(scalar_t));
    std::memcpy(
        col_indices.mutable_data_ptr<HML_INT>(), columns, nnz * sizeof(HML_INT));
  }
  if (rows > 0) {
    std::memcpy(
        crow_indices.mutable_data_ptr<HML_INT>(), rows_start, rows * sizeof(HML_INT));
  }
  crow_indices.mutable_data_ptr<HML_INT>()[rows] = nnz;
}

template <typename scalar_t>
hml_sparse_status_t hml_trsv(
    hml_sparse_operation_t op, scalar_t alpha, hml_sparse_matrix_t A,
    const struct hml_matrix_descr& d, scalar_t* x, scalar_t* y) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return hml_sparse_s_trsv(op, alpha, A, d, x, y);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    return hml_sparse_d_trsv(op, alpha, A, d, x, y);
  } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
    HML_Complex8 ha{alpha.real(), alpha.imag()};
    return hml_sparse_c_trsv(op, ha, A, d, reinterpret_cast<HML_Complex8*>(x), reinterpret_cast<HML_Complex8*>(y));
  } else {
    HML_Complex16 ha{alpha.real(), alpha.imag()};
    return hml_sparse_z_trsv(op, ha, A, d, reinterpret_cast<HML_Complex16*>(x), reinterpret_cast<HML_Complex16*>(y));
  }
}
template <typename scalar_t>
hml_sparse_status_t hml_trsm(
    hml_sparse_operation_t op, scalar_t alpha, hml_sparse_matrix_t A,
    const struct hml_matrix_descr& d, hml_sparse_layout_t layout,
    scalar_t* x, HML_INT columns, HML_INT ldx, scalar_t* y, HML_INT ldy) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return hml_sparse_s_trsm(op, alpha, A, d, layout, x, columns, ldx, y, ldy);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    return hml_sparse_d_trsm(op, alpha, A, d, layout, x, columns, ldx, y, ldy);
  } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
    HML_Complex8 ha{alpha.real(), alpha.imag()};
    return hml_sparse_c_trsm(op, ha, A, d, layout, reinterpret_cast<HML_Complex8*>(x), columns, ldx, reinterpret_cast<HML_Complex8*>(y), ldy);
  } else {
    HML_Complex16 ha{alpha.real(), alpha.imag()};
    return hml_sparse_z_trsm(op, ha, A, d, layout, reinterpret_cast<HML_Complex16*>(x), columns, ldx, reinterpret_cast<HML_Complex16*>(y), ldy);
  }
}
} // namespace

template <typename scalar_t>
hml_sparse_status_t hml_spmmd(
    hml_sparse_operation_t op, hml_sparse_matrix_t A, hml_sparse_matrix_t B,
    hml_sparse_layout_t layout, scalar_t* C, HML_INT ldc) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return hml_sparse_s_spmmd(op, A, B, layout, C, ldc);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    return hml_sparse_d_spmmd(op, A, B, layout, C, ldc);
  } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
    return hml_sparse_c_spmmd(op, A, B, layout, reinterpret_cast<HML_Complex8*>(C), ldc);
  } else {
    return hml_sparse_z_spmmd(op, A, B, layout, reinterpret_cast<HML_Complex16*>(C), ldc);
  }
}
template <typename scalar_t>
hml_sparse_status_t hml_mv(
    hml_sparse_operation_t op, scalar_t alpha, hml_sparse_matrix_t A,
    const struct hml_matrix_descr& d, scalar_t* x, scalar_t beta, scalar_t* y) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return hml_sparse_s_mv(op, alpha, A, d, x, beta, y);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    return hml_sparse_d_mv(op, alpha, A, d, x, beta, y);
  } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
    HML_Complex8 ha{alpha.real(), alpha.imag()}, hb{beta.real(), beta.imag()};
    return hml_sparse_c_mv(op, ha, A, d, reinterpret_cast<HML_Complex8*>(x), hb, reinterpret_cast<HML_Complex8*>(y));
  } else {
    HML_Complex16 ha{alpha.real(), alpha.imag()}, hb{beta.real(), beta.imag()};
    return hml_sparse_z_mv(op, ha, A, d, reinterpret_cast<HML_Complex16*>(x), hb, reinterpret_cast<HML_Complex16*>(y));
  }
}
template <typename scalar_t>
hml_sparse_status_t hml_mm(
    hml_sparse_operation_t op, scalar_t alpha, hml_sparse_matrix_t A,
    const struct hml_matrix_descr& d, hml_sparse_layout_t layout,
    scalar_t* x, HML_INT columns, HML_INT ldx, scalar_t beta, scalar_t* y, HML_INT ldy) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return hml_sparse_s_mm(op, alpha, A, d, layout, x, columns, ldx, beta, y, ldy);
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    return hml_sparse_d_mm(op, alpha, A, d, layout, x, columns, ldx, beta, y, ldy);
  } else if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
    HML_Complex8 ha{alpha.real(), alpha.imag()}, hb{beta.real(), beta.imag()};
    return hml_sparse_c_mm(op, ha, A, d, layout, reinterpret_cast<HML_Complex8*>(x), columns, ldx, hb, reinterpret_cast<HML_Complex8*>(y), ldy);
  } else {
    HML_Complex16 ha{alpha.real(), alpha.imag()}, hb{beta.real(), beta.imag()};
    return hml_sparse_z_mm(op, ha, A, d, layout, reinterpret_cast<HML_Complex16*>(x), columns, ldx, hb, reinterpret_cast<HML_Complex16*>(y), ldy);
  }
}


namespace hml {

inline void triangular_solve_out_sparse_csr(
    const Tensor& A_,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
  if (B.numel() == 0 || X.numel() == 0 || A_._nnz() == 0) {
    X.fill_(NAN);
    return;
  }

  const auto materialize_diagonal_indices = [](const Tensor& t) -> Tensor {
    const auto n = t.size(-1);
    const auto compressed_indices = std::get<0>(at::sparse_csr::getCompressedPlainIndices(t));
    const auto diag_indices = at::arange(n, compressed_indices.options()).unsqueeze(0).expand({2, n});
    const auto diag_values = at::zeros({1}, t.values().options()).expand({n});
    const auto t_coo = t.to_sparse();
    const auto expanded_indices = at::cat({t_coo._indices(), diag_indices}, /*dim=*/-1);
    const auto expanded_values = at::cat({t_coo._values(), diag_values}, /*dim=*/0);
    const auto t_expanded_coo = at::sparse_coo_tensor(expanded_indices, expanded_values, t_coo.sizes(), t_coo.options());
    return t_expanded_coo.to_sparse(t.layout());
  };
  const auto A = unitriangular ? materialize_diagonal_indices(A_) : A_;
  auto X_ = c10::MaybeOwned<Tensor>::owned(X.contiguous());
  auto B_ = c10::MaybeOwned<Tensor>::owned(B.contiguous());
  const bool is_X_row_major = true;
  const bool is_B_row_major = true;
  const hml_sparse_operation_t opA = transpose ? HML_SPARSE_OPERATION_TRANSPOSE : HML_SPARSE_OPERATION_NON_TRANSPOSE;

  struct hml_matrix_descr descrA{};
  descrA.type = HML_SPARSE_MATRIX_TYPE_TRIANGULAR;
  descrA.mode = upper ? HML_SPARSE_FILL_MODE_UPPER : HML_SPARSE_FILL_MODE_LOWER;
  descrA.diag = unitriangular ? HML_SPARSE_DIAG_UNIT : HML_SPARSE_DIAG_NON_UNIT;

  auto A_crow = A.crow_indices().to(at::kInt);
  auto A_col = A.col_indices().to(at::kInt);
  auto A_val = A.values().contiguous();
  const int64_t rows = A.size(0);
  const int64_t cols = A.size(1);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      X.scalar_type(), "triangular_solve_out_sparse_csr_impl_hml", [&] {
        hml_sparse_matrix_t hml_mat;
        auto crow_ptr = A_crow.data_ptr<HML_INT>();
        auto col_ptr = A_col.data_ptr<HML_INT>();
        auto val_ptr = A_val.data_ptr<scalar_t>();
        auto b_ptr = B_->data_ptr<scalar_t>();
        auto x_ptr = X_->data_ptr<scalar_t>();
        hml_sparse_status_t st = hml_create_csr<scalar_t>(
            &hml_mat, rows, cols, crow_ptr, col_ptr, val_ptr);
        TORCH_CHECK(st == 0, "hml_sparse_create_csr failed: ", st);
        scalar_t alpha = 1;
        if (B.size(-1) == 1) {
          st = hml_trsv<scalar_t>(opA, alpha, hml_mat, descrA, b_ptr, x_ptr);
        } else {
          auto order = HML_SPARSE_LAYOUT_ROW_MAJOR;
          auto nrhs = static_cast<HML_INT>(B.size(-1));
          auto ldx = static_cast<HML_INT>(X.size(1));
          auto ldb = static_cast<HML_INT>(B.size(1));
          st = hml_trsm<scalar_t>(opA, alpha, hml_mat, descrA, order, b_ptr, nrhs, ldb, x_ptr, ldx);
        }
        hml_sparse_destroy(hml_mat);
        TORCH_CHECK(st == 0, "hml_sparse trsm/trsv failed: ", st);
      });
}

inline void add_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& alpha,
    const Tensor& result) {
  // HSPARSE doesn't work with empty matrices
  if (mat2._nnz() == 0) {
    hml_col_indices_and_values_resize_(result, mat1._nnz());
    result.copy_(mat1);
    return;
  } else if (mat1._nnz() == 0) {
    hml_col_indices_and_values_resize_(result, mat2._nnz());
    result.copy_(mat2);
    result.values().mul_(alpha);
    return;
  }

  const auto output_indices_dtype = promoteTypes(mat1.crow_indices().scalar_type(), mat2.crow_indices().scalar_type());
  auto result_crow_indices_backup = result.crow_indices();
  auto result_col_indices_backup = result.col_indices();
  hml_indices_to_compatible_inplace(result);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "add_out_sparse_csr_impl_hml", [&] {
        auto alpha_ = alpha.to<scalar_t>();
        // C = alpha * mat2 + mat1（与 MKL 相同：pytorch 需 mat1 + alpha*mat2）
        auto A_crow = mat1.crow_indices().to(at::kInt);
        auto A_col = mat1.col_indices().to(at::kInt);
        auto A_val = mat1.values().contiguous();
        auto B_crow = mat2.crow_indices().to(at::kInt);
        auto B_col = mat2.col_indices().to(at::kInt);
        auto B_val = mat2.values().contiguous();
        hml_sparse_matrix_t hml_A, hml_B, hml_C;
        TORCH_CHECK(hml_create_csr<scalar_t>(
                        &hml_A, mat1.size(0), mat1.size(1), A_crow.data_ptr<HML_INT>(),
                        A_col.data_ptr<HML_INT>(), A_val.data_ptr<scalar_t>()) == 0,
                    "hml_sparse create_csr A failed");
        TORCH_CHECK(hml_create_csr<scalar_t>(
                        &hml_B, mat2.size(0), mat2.size(1), B_crow.data_ptr<HML_INT>(),
                        B_col.data_ptr<HML_INT>(), B_val.data_ptr<scalar_t>()) == 0,
                    "hml_sparse create_csr B failed");
        hml_sparse_status_t st = hml_add<scalar_t>(
            HML_SPARSE_OPERATION_NON_TRANSPOSE, alpha_, hml_B, hml_A, &hml_C);
        TORCH_CHECK(st == 0, "hml_sparse add failed: ", st);
        hml_result_copy_<scalar_t>(result, hml_C);
        hml_sparse_destroy(hml_A);
        hml_sparse_destroy(hml_B);
        hml_sparse_destroy(hml_C);
      });

  if (output_indices_dtype == at::kLong) {
    const auto res_nnz = result._nnz();
    static_cast<SparseCsrTensorImpl*>(result.unsafeGetTensorImpl())->set_member_tensors(
        result_crow_indices_backup.copy_(result.crow_indices()),
        result_col_indices_backup.resize_({res_nnz}).copy_(result.col_indices()),
        result.values(),
        result.sizes());
  }
}

inline void addmv_out_sparse_csr(
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  TORCH_CHECK(mat.layout() == kSparseCsr, "HSPARSE supports CSR layout only");
  c10::MaybeOwned<Tensor> result_ = c10::MaybeOwned<Tensor>::owned(result.contiguous());
  c10::MaybeOwned<Tensor> vec_ = c10::MaybeOwned<Tensor>::owned(vec.contiguous());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "addmv_out_sparse_csr_impl_hml", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        auto A_crow = mat.crow_indices().to(at::kInt);
        auto A_col = mat.col_indices().to(at::kInt);
        auto A_val = mat.values().contiguous();
        struct hml_matrix_descr descrA{};
        descrA.type = HML_SPARSE_MATRIX_TYPE_GENERAL;
        hml_sparse_matrix_t hml_mat;
        auto crow_ptr = A_crow.data_ptr<HML_INT>();
        hml_sparse_status_t st = hml_create_csr<scalar_t>(
            &hml_mat, mat.size(0), mat.size(1), crow_ptr,
            A_col.data_ptr<HML_INT>(), A_val.data_ptr<scalar_t>());
        TORCH_CHECK(st == 0, "hml_sparse create_csr failed: ", st);
        st = hml_mv<scalar_t>(
            HML_SPARSE_OPERATION_NON_TRANSPOSE, alpha_, hml_mat, descrA,
            vec_->data_ptr<scalar_t>(), beta_, result_->data_ptr<scalar_t>());
        hml_sparse_destroy(hml_mat);
        TORCH_CHECK(st == 0, "hml_sparse mv failed: ", st);
      });
}

inline void addmm_sparse_input_dense_result(
    const Tensor& A,
    const Tensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& C) {
  TORCH_CHECK(A.layout() == kSparseCsr && B.layout() == kSparseCsr, "HSPARSE supports CSR layout only");
  auto C_ = at::empty(C.sizes(), C.options());
  auto order = HML_SPARSE_LAYOUT_ROW_MAJOR;
  auto ldc = static_cast<HML_INT>(C_.stride(-2));
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      C.scalar_type(), "addmm_sparse_input_dense_result_hml", [&] {
        auto A_crow = A.crow_indices().to(at::kInt);
        auto A_col = A.col_indices().to(at::kInt);
        auto A_val = A.values().contiguous();
        auto B_crow = B.crow_indices().to(at::kInt);
        auto B_col = B.col_indices().to(at::kInt);
        auto B_val = B.values().contiguous();
        hml_sparse_matrix_t hml_A, hml_B;
        hml_sparse_status_t st = hml_create_csr<scalar_t>(
            &hml_A, A.size(0), A.size(1), A_crow.data_ptr<HML_INT>(),
            A_col.data_ptr<HML_INT>(), A_val.data_ptr<scalar_t>());
        TORCH_CHECK(st == 0, "hml create_csr A failed: ", st);
        st = hml_create_csr<scalar_t>(
            &hml_B, B.size(0), B.size(1), B_crow.data_ptr<HML_INT>(),
            B_col.data_ptr<HML_INT>(), B_val.data_ptr<scalar_t>());
        TORCH_CHECK(st == 0, "hml create_csr B failed: ", st);
        st = hml_spmmd<scalar_t>(
            HML_SPARSE_OPERATION_NON_TRANSPOSE, hml_A, hml_B, order,
            C_.data_ptr<scalar_t>(), ldc);
        hml_sparse_destroy(hml_A);
        hml_sparse_destroy(hml_B);
        TORCH_CHECK(st == 0, "hml_sparse spmmd failed: ", st);
      });
  if (beta.toComplexDouble() == 0.) {
    C.zero_();
  } else {
    C.mul_(beta);
  }
  C.add_(C_, alpha);
}

inline void addmm_dense_result(
    const Tensor& A,
    const Tensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& C) {
  TORCH_CHECK(A.layout() == kSparseCsr, "HSPARSE supports CSR layout only");
  c10::MaybeOwned<Tensor> C_ = c10::MaybeOwned<Tensor>::owned(C.contiguous());
  c10::MaybeOwned<Tensor> B_ = c10::MaybeOwned<Tensor>::owned(B.contiguous());
  auto ldc = static_cast<HML_INT>(C.size(1));
  auto ldb = static_cast<HML_INT>(B.size(1));
  auto columns_C = static_cast<HML_INT>(C.size(-1));
  struct hml_matrix_descr descrA{};
  descrA.type = HML_SPARSE_MATRIX_TYPE_GENERAL;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      C.scalar_type(), "addmm_out_sparse_csr_impl_hml", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        auto A_crow = A.crow_indices().to(at::kInt);
        auto A_col = A.col_indices().to(at::kInt);
        auto A_val = A.values().contiguous();
        hml_sparse_matrix_t hml_mat;
        auto crow_ptr = A_crow.data_ptr<HML_INT>();
        hml_sparse_status_t st = hml_create_csr<scalar_t>(
            &hml_mat, A.size(0), A.size(1), crow_ptr,
            A_col.data_ptr<HML_INT>(), A_val.data_ptr<scalar_t>());
        TORCH_CHECK(st == 0, "hml_sparse create_csr failed: ", st);
        st = hml_mm<scalar_t>(
            HML_SPARSE_OPERATION_NON_TRANSPOSE, alpha_, hml_mat, descrA,
            HML_SPARSE_LAYOUT_ROW_MAJOR,
            B_->data_ptr<scalar_t>(), columns_C, ldb, beta_,
            C_->data_ptr<scalar_t>(), ldc);
        hml_sparse_destroy(hml_mat);
        TORCH_CHECK(st == 0, "hml_sparse mm failed: ", st);
      });
  if (!C.is_same(*C_)) {
    C.copy_(*C_);
  }
}

} // namespace hml
