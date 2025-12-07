#[cfg(feature = "matrix")]
use ndarray_linalg::Inverse;

use chili_core::{SpicyError, SpicyObj, SpicyResult};

#[cfg(feature = "matrix")]
pub fn inv(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    match arg0 {
        SpicyObj::Matrix(m) => Ok(SpicyObj::Matrix(
            m.inv()
                .map_err(|e| SpicyError::Err(e.to_string()))?
                .to_shared(),
        )),
        _ => Err(SpicyError::Err(format!(
            "requires matrix, got '{}'",
            arg0.get_type_name()
        ))),
    }
}

#[cfg(not(feature = "matrix"))]
pub fn inv(_: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    Err(SpicyError::Err(
        "matrix operations are not enabled".to_string(),
    ))
}
