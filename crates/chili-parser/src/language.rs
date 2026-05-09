#[derive(Debug, Clone, PartialEq, Default, Copy)]
#[repr(u8)]
pub enum Language {
    #[default]
    Chili,
    Pepper,
}

impl Language {
    pub fn from_extension(extension: &str) -> Option<Self> {
        if extension == "chi" {
            Some(Self::Chili)
        } else if extension == "pep" {
            Some(Self::Pepper)
        } else {
            None
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Chili => "chi",
            Self::Pepper => "pep",
        }
    }
}

impl TryFrom<u8> for Language {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Chili),
            1 => Ok(Self::Pepper),
            _ => Err(format!("invalid language value: {}", value)),
        }
    }
}
