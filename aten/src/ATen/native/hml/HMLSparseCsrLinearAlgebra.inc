class SparseCsrHMLInterface {
 private:
  hml_sparse_matrix_t A{nullptr};
  struct hml_matrix_descr desc;
 public:
  SparseCsrHMLInterface(
      HML_INT* col_indices,
      HML_INT* crow_indices,
      double* values,
      HML_INT nrows,
      HML_INT ncols,
      c10::Layout layout = kSparseCsr) {
    TORCH_CHECK(layout == kSparseCsr, "HSPARSE only supports CSR layout");

    desc.type = HML_SPARSE_MATRIX_TYPE_GENERAL;
    int retval = hml_sparse_d_create_csr(
        &A,
        HML_SPARSE_INDEX_BASE_ZERO,
        nrows,
        ncols,
        crow_indices,
        crow_indices + 1,
        col_indices,
        values);
    TORCH_CHECK(
        retval == 0,
        "hml_sparse_d_create_csr failed with error code: ",
        retval);
  }
  SparseCsrHMLInterface(
      HML_INT* col_indices,
      HML_INT* crow_indices,
      float* values,
      HML_INT nrows,
      HML_INT ncols) {
    desc.type = HML_SPARSE_MATRIX_TYPE_GENERAL;
    int retval = hml_sparse_s_create_csr(
        &A,
        HML_SPARSE_INDEX_BASE_ZERO,
        nrows,
        ncols,
        crow_indices,
        crow_indices + 1,
        col_indices,
        values);
    TORCH_CHECK(
        retval == 0,
        "hml_sparse_s_create_csr failed with error code: ",
        retval);
  }
  inline void sparse_mm(
      double* res,
      double* dense,
      double alpha,
      double beta,
      HML_INT nrows,
      HML_INT ncols,
      HML_INT dense_ncols) {
    int stat;
    if (dense_ncols == 1) {
      stat = hml_sparse_d_mv(
        HML_SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A,
        desc,
        dense,
        beta,
        res);
      TORCH_CHECK(stat == 0, "hml_sparse_d_mv failed with error code: ", stat);
    } else {
      stat = hml_sparse_d_mm(
        HML_SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A,
        desc,
        HML_SPARSE_LAYOUT_ROW_MAJOR,
        dense,
        nrows,
        ncols,
        beta,
        res,
        dense_ncols);
      TORCH_CHECK(stat == 0, "hml_sparse_d_mm failed with error code: ", stat);
    }
  }
  inline void sparse_mm(
      float* res,
      float* dense,
      float alpha,
      float beta,
      HML_INT nrows,
      HML_INT ncols,
      HML_INT dense_ncols) {
    int stat;
    if (dense_ncols == 1) {
      stat = hml_sparse_s_mv(
        HML_SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A,
        desc,
        dense,
        beta,
        res);
      TORCH_CHECK(stat == 0, "hml_sparse_s_mv failed with error code: ", stat);
    } else {
      stat = hml_sparse_s_mm(
        HML_SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A,
        desc,
        HML_SPARSE_LAYOUT_ROW_MAJOR,
        dense,
        nrows,
        ncols,
        beta,
        res,
        dense_ncols);
      TORCH_CHECK(stat == 0, "hml_sparse_s_mm failed with error code: ", stat);
    }
  }
  ~SparseCsrHMLInterface() {
    hml_sparse_destroy(A);
  }
};
