//! Aligned with tig-challenges from tig-foundation/tig-monorepo (c008_energy_arbitrage).
//! See: https://github.com/tig-foundation/tig-monorepo/blob/c008_energy_arbitrage/tig-challenges/src/lib.rs

#![allow(unused_imports, dead_code)]

pub const BUILD_TIME_PATH: &str = env!("CARGO_MANIFEST_DIR");

const QUALITY_PRECISION: i32 = 1_000_000;

macro_rules! conditional_pub {
    (fn $name:ident $($rest:tt)*) => {
        #[cfg(not(feature = "hide_verification"))]
        pub fn $name $($rest)*

        #[cfg(feature = "hide_verification")]
        fn $name $($rest)*
    };
}

macro_rules! impl_kv_string_serde {
    ($name:ident { $( $field:ident : $ty:ty ),* $(,)? }) => {
        paste::paste! {
            #[derive(Debug, Clone, PartialEq)]
            pub struct $name {
                $( pub $field : $ty ),*
            }

            impl serde::Serialize for $name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    let mut parts = Vec::new();
                    $(
                        parts.push(format!("{}={}", stringify!($field), self.$field));
                    )*
                    parts.sort();
                    let s = parts.join(",");
                    serializer.serialize_str(&s)
                }
            }

            impl<'de> serde::Deserialize<'de> for $name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de>
                {
                    use serde::de::{Visitor, Error};
                    use std::fmt;

                    struct VisitorImpl;

                    impl<'de> Visitor<'de> for VisitorImpl {
                        type Value = $name;

                        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            write!(f, "a string of the form 'key=value,key=value'")
                        }

                        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                        where
                            E: Error,
                        {
                            let mut map = std::collections::HashMap::new();

                            if !v.is_empty() {
                                for part in v.split(',') {
                                    let mut kv = part.splitn(2, '=');
                                    let key = kv.next().ok_or_else(|| E::custom(format!("Missing key in '{}'", part)))?;
                                    let val = kv.next().ok_or_else(|| E::custom(format!("Missing value in '{}'", part)))?;
                                    map.insert(key, val);
                                }
                            }

                            Ok($name {
                                $(
                                    $field: map.get(stringify!($field))
                                        .ok_or_else(|| E::custom(format!("Missing field '{}'", stringify!($field))))?
                                        .parse::<$ty>()
                                        .map_err(E::custom)?,
                                )*
                            })
                        }
                    }

                    deserializer.deserialize_str(VisitorImpl)
                }
            }
        }
    };
}

macro_rules! impl_base64_serde {
    ($name:ident { $( $field:ident : $ty:ty ),* $(,)? }) => {
        paste::paste! {
            #[derive(Debug, Clone)]
            pub struct $name {
                $( pub $field : $ty ),*
            }

            #[derive(serde::Serialize, serde::Deserialize)]
            struct [<$name Data>] {
                $( $field : $ty ),*
            }

            impl serde::Serialize for $name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    use flate2::write::GzEncoder;
                    use flate2::Compression;
                    use base64::engine::general_purpose::STANDARD as BASE64;
                    use base64::Engine;
                    use std::io::Write;

                    let helper = [<$name Data>] {
                        $( $field: self.$field.clone() ),*
                    };

                    let bincode_data = bincode::serialize(&helper)
                        .map_err(|e| serde::ser::Error::custom(format!("Bincode serialization failed: {}", e)))?;

                    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                    encoder
                        .write_all(&bincode_data)
                        .map_err(|e| serde::ser::Error::custom(format!("Compression failed: {}", e)))?;
                    let compressed_data = encoder
                        .finish()
                        .map_err(|e| serde::ser::Error::custom(format!("Compression finish failed: {}", e)))?;

                    let encoded = BASE64.encode(&compressed_data);
                    serializer.serialize_str(&encoded)
                }
            }

            impl<'de> serde::Deserialize<'de> for $name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de>,
                {
                    use flate2::read::GzDecoder;
                    use base64::engine::general_purpose::STANDARD as BASE64;
                    use base64::Engine;
                    use std::io::Read;
                    use std::fmt;

                    struct VisitorImpl;

                    impl<'de> serde::de::Visitor<'de> for VisitorImpl {
                        type Value = $name;

                        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            write!(f, "a base64 encoded, compressed, bincode serialized {}", stringify!($name))
                        }

                        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                        where
                            E: serde::de::Error,
                        {
                            let compressed = BASE64.decode(v)
                                .map_err(|e| E::custom(format!("Base64 decode failed: {}", e)))?;

                            let mut decoder = GzDecoder::new(&compressed[..]);
                            let mut decompressed = Vec::new();
                            decoder
                                .read_to_end(&mut decompressed)
                                .map_err(|e| E::custom(format!("Decompression failed: {}", e)))?;

                            let data: [<$name Data>] = bincode::deserialize(&decompressed)
                                .map_err(|e| E::custom(format!("Bincode deserialization failed: {}", e)))?;

                            Ok($name {
                                $( $field: data.$field ),*
                            })
                        }
                    }

                    deserializer.deserialize_str(VisitorImpl)
                }
            }
        }
    };
}

#[cfg(feature = "c008")]
pub mod energy_arbitrage;
#[cfg(feature = "c008")]
pub use energy_arbitrage as c008;
