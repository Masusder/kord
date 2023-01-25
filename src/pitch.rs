// Traits.

use once_cell::sync::Lazy;

/// A trait for types that have a pitch property.
pub trait HasPitch {
    /// Returns the pitch of the type (usually a [`Note`]).
    fn pitch(&self) -> Pitch;
}

/// A trait for types that have a base frequency property.
pub trait HasBaseFrequency {
    /// Returns the base frequency of the type (usually a [`Pitch`]).
    fn base_frequency(&self) -> f32;
}

/// A trait for types that have a frequency property.
pub trait HasFrequency {
    /// Returns the frequency of the type (usually a [`Note`]).
    fn frequency(&self) -> f32;
}

// Enum.

/// An enum representing the pitch of a note.
///
/// The frequencies of the pitches are based on the [A4 frequency](https://en.wikipedia.org/wiki/A4_(pitch_standard)).
/// There is no enharmonic representation here, so all of the sharps are represented.
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Ord, PartialOrd)]
#[repr(u8)]
pub enum Pitch {
    /// The pitch C.
    C,
    /// The pitch C♯.
    DFlat,
    /// The pitch D.
    D,
    /// The pitch D♯.
    EFlat,
    /// The pitch E.
    E,
    /// The pitch F.
    F,
    /// The pitch F♯.
    GFlat,
    /// The pitch G.
    G,
    /// The pitch G♯.
    AFlat,
    /// The pitch A.
    A,
    /// The pitch A♯.
    BFlat,
    /// The pitch B.
    B,
}

// Pitch impls.

impl HasBaseFrequency for Pitch {
    #[no_coverage]
    fn base_frequency(&self) -> f32 {
        match self {
            Pitch::C => 16.35,
            Pitch::DFlat => 17.32,
            Pitch::D => 18.35,
            Pitch::EFlat => 19.45,
            Pitch::E => 20.60,
            Pitch::F => 21.83,
            Pitch::GFlat => 23.12,
            Pitch::G => 24.50,
            Pitch::AFlat => 25.96,
            Pitch::A => 27.50,
            Pitch::BFlat => 29.14,
            Pitch::B => 30.87,
        }
    }
}

impl HasPitch for Pitch {
    fn pitch(&self) -> Pitch {
        *self
    }
}

impl TryFrom<u8> for Pitch {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Pitch::C),
            1 => Ok(Pitch::DFlat),
            2 => Ok(Pitch::D),
            3 => Ok(Pitch::EFlat),
            4 => Ok(Pitch::E),
            5 => Ok(Pitch::F),
            6 => Ok(Pitch::GFlat),
            7 => Ok(Pitch::G),
            8 => Ok(Pitch::AFlat),
            9 => Ok(Pitch::A),
            10 => Ok(Pitch::BFlat),
            11 => Ok(Pitch::B),
            _ => Err("Invalid pitch"),
        }
    }
}

// Statics.

pub(crate) static ALL_PITCHES: Lazy<[Pitch; 12]> = Lazy::new(|| {
    [
        Pitch::C,
        Pitch::DFlat,
        Pitch::D,
        Pitch::EFlat,
        Pitch::E,
        Pitch::F,
        Pitch::GFlat,
        Pitch::G,
        Pitch::AFlat,
        Pitch::A,
        Pitch::BFlat,
        Pitch::B,
    ]
});

// Tests.

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_properties() {
        assert_eq!(Pitch::G.pitch(), Pitch::G);
        assert_eq!(Pitch::G.base_frequency(), 24.50);
    }
}
