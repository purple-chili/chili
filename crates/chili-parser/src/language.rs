#[derive(Debug, Clone, PartialEq, Default)]
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
