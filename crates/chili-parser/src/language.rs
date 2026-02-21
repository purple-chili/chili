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

impl From<u8> for Language {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Chili,
            1 => Self::Pepper,
            _ => panic!("invalid language value: {}", value),
        }
    }
}
