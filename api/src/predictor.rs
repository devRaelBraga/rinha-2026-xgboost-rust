use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ffi::CString;
use std::sync::mpsc;
use tokio::sync::oneshot;

pub type DMatrixHandle = *mut c_void;
pub type BoosterHandle = *mut c_void;
pub type bst_ulong = u64;

#[link(name = "xgboost")]
extern "C" {
    fn XGDMatrixCreateFromMat(
        data: *const f32,
        nrow: bst_ulong,
        ncol: bst_ulong,
        missing: f32,
        out: *mut DMatrixHandle,
    ) -> c_int;

    fn XGDMatrixFree(handle: DMatrixHandle) -> c_int;

    fn XGBoosterCreate(
        dmats: *const DMatrixHandle,
        len: bst_ulong,
        out: *mut BoosterHandle,
    ) -> c_int;

    fn XGBoosterFree(handle: BoosterHandle) -> c_int;

    fn XGBoosterLoadModel(handle: BoosterHandle, fname: *const c_char) -> c_int;

    fn XGBoosterSetParam(
        handle: BoosterHandle,
        name: *const c_char,
        value: *const c_char,
    ) -> c_int;

    fn XGBoosterPredict(
        handle: BoosterHandle,
        dmat: DMatrixHandle,
        option_mask: c_int,
        ntree_limit: c_uint,
        training: c_int,
        out_len: *mut bst_ulong,
        out_result: *mut *const f32,
    ) -> c_int;
}

struct SyncBooster(BoosterHandle);
unsafe impl Send for SyncBooster {}

pub struct Predictor {
    tx: mpsc::SyncSender<( [f32; 14], oneshot::Sender<f64> )>,
}

impl Predictor {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let mut handle: BoosterHandle = std::ptr::null_mut();
            if XGBoosterCreate(std::ptr::null(), 0, &mut handle) != 0 {
                return Err("XGBoosterCreate failed".into());
            }

            let c_path = CString::new(model_path)?;
            if XGBoosterLoadModel(handle, c_path.as_ptr()) != 0 {
                return Err("XGBoosterLoadModel failed".into());
            }

            let p_key = CString::new("nthread")?;
            let p_val = CString::new("1")?;
            XGBoosterSetParam(handle, p_key.as_ptr(), p_val.as_ptr());

            let handle_ptr = handle as usize;
            let (tx, rx) = mpsc::sync_channel::<( [f32; 14], oneshot::Sender<f64> )>(1024);

            std::thread::spawn(move || {
                let handle = handle_ptr as BoosterHandle;
                for (features, resp) in rx {
                    let mut dmatrix: DMatrixHandle = std::ptr::null_mut();
                    let mut out_score = 0.0;

                    if XGDMatrixCreateFromMat(features.as_ptr(), 1, 14, 0.0, &mut dmatrix) == 0 {
                        let mut out_len: bst_ulong = 0;
                        let mut out_ptr: *const f32 = std::ptr::null();
                        
                        if XGBoosterPredict(handle, dmatrix, 0, 0, 0, &mut out_len, &mut out_ptr) == 0 {
                            if out_len > 0 && !out_ptr.is_null() {
                                out_score = *out_ptr as f64;
                                if out_score < 0.0 { out_score = 0.0; }
                                if out_score > 1.0 { out_score = 1.0; }
                            }
                        }
                        XGDMatrixFree(dmatrix);
                    }
                    
                    let _ = resp.send(out_score);
                }
                XGBoosterFree(handle);
            });

            Ok(Self { tx })
        }
    }

    pub async fn predict(&self, features: [f32; 14]) -> Option<f64> {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self.tx.try_send((features, resp_tx)).is_ok() {
            if let Ok(Ok(score)) = tokio::time::timeout(std::time::Duration::from_millis(50), resp_rx).await {
                return Some(score);
            }
        }
        None
    }
}
