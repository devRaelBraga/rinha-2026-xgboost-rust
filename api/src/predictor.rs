use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ffi::CString;
pub type DMatrixHandle = *mut c_void;
pub type BoosterHandle = *mut c_void;
pub type bst_ulong = u64;

#[link(name = "xgboost")]
extern "C" {
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

    fn XGBoosterPredictFromDense(
        handle: BoosterHandle,
        values: *const c_char,
        config: *const c_char,
        m: DMatrixHandle,
        out_shape: *mut *const bst_ulong,
        out_dim: *mut bst_ulong,
        out_result: *mut *const f32,
    ) -> c_int;
}

pub struct Predictor {
    handle: BoosterHandle,
}

unsafe impl Send for Predictor {}
unsafe impl Sync for Predictor {}

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

            Ok(Self { handle })
        }
    }
    #[inline(always)]
    pub fn predict(&self, features: &[f32; 14]) -> Option<f64> {
        unsafe {
            let ptr = features.as_ptr() as u64;
            
            const JSON_PREFIX: &[u8] = b"{\"data\": [";
            const JSON_SUFFIX: &[u8] = b", false], \"shape\": [1, 14], \"typestr\": \"<f4\", \"version\": 3}\0";
            
            let mut buf = [0u8; 128];
            let mut idx = 0;
            
            buf[idx..idx + JSON_PREFIX.len()].copy_from_slice(JSON_PREFIX);
            idx += JSON_PREFIX.len();
            
            if ptr == 0 {
                buf[idx] = b'0';
                idx += 1;
            } else {
                const PAIRS: &[u8; 200] = b"00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
                let mut temp = [0u8; 20];
                let mut t_idx = 20;
                let mut n = ptr;
                
                while n >= 100 {
                    let r = (n % 100) as usize * 2;
                    n /= 100;
                    t_idx -= 2;
                    temp[t_idx] = PAIRS[r];
                    temp[t_idx + 1] = PAIRS[r + 1];
                }
                
                if n < 10 {
                    t_idx -= 1;
                    temp[t_idx] = b'0' + n as u8;
                } else {
                    let r = n as usize * 2;
                    t_idx -= 2;
                    temp[t_idx] = PAIRS[r];
                    temp[t_idx + 1] = PAIRS[r + 1];
                }
                
                let len = 20 - t_idx;
                buf[idx..idx + len].copy_from_slice(&temp[t_idx..20]);
                idx += len;
            }
            
            buf[idx..idx + JSON_SUFFIX.len()].copy_from_slice(JSON_SUFFIX);
            
            const CONFIG_JSON: &[u8] = b"{\"type\": 1, \"missing\": 0.0, \"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}\0";

            let mut out_shape: *const bst_ulong = std::ptr::null();
            let mut out_dim: bst_ulong = 0;
            let mut out_result: *const f32 = std::ptr::null();

            if XGBoosterPredictFromDense(
                self.handle, 
                buf.as_ptr() as *const c_char, 
                CONFIG_JSON.as_ptr() as *const c_char, 
                std::ptr::null_mut(), 
                &mut out_shape, 
                &mut out_dim, 
                &mut out_result
            ) == 0 {
                if !out_result.is_null() {
                    return Some((*out_result as f64).clamp(0.0, 1.0));
                }
            }
            None
        }
    }
}

impl Drop for Predictor {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                XGBoosterFree(self.handle);
            }
        }
    }
}
