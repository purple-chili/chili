#[derive(Debug)]
pub struct AuthInfo {
    pub username: String,
    pub is_authenticated: bool,
    pub version: u8,
}
