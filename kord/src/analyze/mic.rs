//! Analyzes audio data from the microphone.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::Context;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    InputCallbackInfo,
};

use crate::core::{base::Res, note::Note};

use super::base::get_notes_from_audio_data;

/// Gets notes from the microphone input over the specified period of time.
pub async fn get_notes_from_microphone(length_in_seconds: f32) -> Res<Vec<Note>> {
    // Get data.

    let data_from_microphone = get_audio_data_from_microphone(length_in_seconds).await?;

    // Get notes.

    let result = get_notes_from_audio_data(&data_from_microphone, length_in_seconds)?;

    Ok(result)
}

/// Gets audio data from the microphone.
pub async fn get_audio_data_from_microphone(length_in_seconds: f32) -> Res<Vec<f32>> {
    if length_in_seconds < 0.2 {
        return Err(anyhow::Error::msg("Listening length in seconds must be greater than 0.2."));
    }

    // Set up devices and systems.

    let (device, config) = get_device_and_config()?;

    // Record audio from the microphone.

    let data_from_microphone = record_from_device(device, config, length_in_seconds).await?;

    Ok(data_from_microphone)
}

/// Gets the system device, and config.
fn get_device_and_config() -> Res<(cpal::Device, cpal::SupportedStreamConfig)> {
    let host = cpal::default_host();

    let device = host.default_input_device().ok_or_else(|| anyhow::Error::msg("Failed to get default input device."))?;

    let config = device.default_input_config().context("Could not get default input config.")?;

    Ok((device, config))
}

/// Records audio data from the device.
async fn record_from_device(device: cpal::Device, config: cpal::SupportedStreamConfig, length_in_seconds: f32) -> Res<Vec<f32>> {
    // Set up recording.

    let sample_rate = config.sample_rate().0 as f32;
    let channels = config.channels() as usize;
    let required_samples = (sample_rate * length_in_seconds) as usize * channels;

    // let likely_sample_count = config.sample_rate().0 as f32 * config.channels() as f32 * length_in_seconds;

    let data_from_microphone = Arc::new(Mutex::new(Vec::with_capacity(required_samples)));

    // let data_from_microphone = Arc::new(Mutex::new(Vec::with_capacity(likely_sample_count as usize)));
    let last_error = Arc::new(Mutex::new(None));

    let stream = {
        let result = data_from_microphone.clone();
        let last_error = last_error.clone();

        device.build_input_stream::<f32, _, _>(
            &config.into(),
            move |data: &[_], _: &InputCallbackInfo| {
                result.lock().unwrap().extend_from_slice(data);
            },
            move |err| {
                last_error.lock().unwrap().replace(err);
            },
            None,
        )?
    };

    // Begin recording.

    stream.play()?;
    futures_timer::Delay::new(Duration::from_secs_f32(length_in_seconds)).await;
    drop(stream);

    // SAFETY: We are the only thread that can access the arc right now since the stream is dropped.
    if let Err(err) = Arc::try_unwrap(last_error).unwrap().into_inner() {
        return Err(err.into());
    }

    // SAFETY: We are the only thread that can access the arc right now since the stream is dropped.
    let data_from_microphone = Arc::try_unwrap(data_from_microphone).unwrap().into_inner()?;

    Ok(data_from_microphone)
}

// Tests.

#[cfg(test)]
mod tests {
    use rand::{seq::SliceRandom, thread_rng};
    use std::f32::consts::PI;
    use std::sync::LazyLock;

    use crate::{
        core::base::HasName,
        core::note::ALL_PITCH_NOTES,
        core::pitch::HasFrequency,
        core::{base::Parsable, chord::Chord, note::Note},
    };

    static REALTIME_DURATION: f32 = 0.2;
    static VALID_NOTES_GUITAR: LazyLock<Vec<Note>> = LazyLock::new(|| {
        ALL_PITCH_NOTES
            .iter()
            .copied()
            .filter(|note| (80.0..=8_000.0).contains(&note.frequency()))
            .collect()
    });

    fn generate_test_tone(duration: f32, frequencies: &[f32]) -> Vec<f32> {
        static SAMPLE_RATE: u32 = 44100;
        let sample_count = (duration * SAMPLE_RATE as f32) as usize;
        let amplitude = 0.5 / frequencies.len() as f32;

        (0..sample_count)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                frequencies.iter().fold(0.0, |acc, &freq| acc + (2.0 * PI * freq * t).sin() * amplitude)
            })
            .collect()
    }

    #[test]
    fn test_mic() {
        let data = crate::analyze::base::tests::load_test_data();

        let notes = Note::try_from_audio(&data, 5.0).unwrap();

        let chord = Chord::try_from_notes(&notes).unwrap();

        assert_eq!(chord[0], Chord::parse("C7b9").unwrap());
    }

    #[test]
    fn test_single_note_realtime() {
        let mut rng = thread_rng();

        let selected_note = *VALID_NOTES_GUITAR.choose(&mut rng).unwrap();
        let frequency = selected_note.frequency();

        let data = generate_test_tone(REALTIME_DURATION, &[frequency]);

        let notes = Note::try_from_audio(&data, REALTIME_DURATION).unwrap();

        assert_eq!(notes.len(), 1, "Expected exactly one note (got {})", notes.len());
        assert_eq!(notes[0], selected_note, "Detected note '{}' doesn't match expected '{}'", notes[0].name(), selected_note.name());
    }

    #[test]
    fn test_multiple_notes_realtime() {
        let mut rng = thread_rng();

        let selected_notes = VALID_NOTES_GUITAR.choose_multiple(&mut rng, 3).copied().collect::<Vec<_>>();

        let frequencies: Vec<f32> = selected_notes.iter().map(|n| n.frequency()).collect();

        let data = generate_test_tone(REALTIME_DURATION, &frequencies);

        let detected_notes = Note::try_from_audio(&data, REALTIME_DURATION).unwrap();

        let valid_count = detected_notes.iter().filter(|n| selected_notes.contains(n)).count();

        assert!(detected_notes.len() >= 2, "Should detect at least 2 notes (got {})", detected_notes.len());
        assert!(
            valid_count >= 2,
            "At least 2 detected notes must match expected.\n\
             Selected Notes: {:?}\n\
             Detected Notes: {:?}",
            selected_notes.iter().map(|n| n.name()).collect::<Vec<_>>(),
            detected_notes.iter().map(|n| n.name()).collect::<Vec<_>>()
        );
    }
}
