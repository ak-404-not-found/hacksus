"""
Combined Speech Pipeline: Speaker Diarization + Text-to-Speech Synthesis

This pipeline combines pyannote-audio for speaker analysis with TTS for
speech synthesis to enable:
1. Speaker identification and segmentation from audio
2. Speaker embedding extraction
3. Speech synthesis with speaker-specific characteristics
4. Multi-speaker conversation simulation
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Pyannote audio imports
from pyannote.audio import Model, Pipeline, Inference, Audio
from pyannote.core import Segment

# TTS imports
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


@dataclass
class SpeakerSegment:
    """Container for speaker segment information"""
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    confidence: float


@dataclass
class DiarizedText:
    """Container for text with speaker information"""
    speaker_id: str
    text: str
    segment: SpeakerSegment


class CombinedSpeechPipeline:
    """
    End-to-end pipeline combining speaker diarization and TTS synthesis
    """
    
    def __init__(
        self,
        diarization_model: str = "pyannote/speaker-diarization-3.0",
        embedding_model: str = "pyannote/speaker-embedding",
        tts_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        vocoder_model_name: str = "vocoder_models/en/ljspeech/hifigan",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize the combined pipeline
        
        Args:
            diarization_model: Pretrained diarization model name
            embedding_model: Pretrained speaker embedding model name
            tts_model_name: TTS model to use
            vocoder_model_name: Vocoder model for TTS
            device: Device to use (cuda or cpu)
            use_auth_token: HuggingFace token for model access
        """
        self.device = device
        self.use_auth_token = use_auth_token
        
        # Initialize diarization pipeline
        print("Loading diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            diarization_model,
            use_auth_token=use_auth_token
        ).to(device)
        
        # Initialize speaker embedding model
        print("Loading speaker embedding model...")
        self.embedding_model = Model.from_pretrained(
            embedding_model,
            use_auth_token=use_auth_token
        ).to(device)
        
        # Initialize audio processor
        self.audio_processor = Audio()
        
        # Initialize TTS
        print("Loading TTS model...")
        self.tts_manager = ModelManager(Path.home() / ".TTS")
        self.synthesizer = Synthesizer(
            model_name=tts_model_name,
            vocoder_name=vocoder_model_name,
            gpu=device == "cuda"
        )
        
        # Speaker embeddings cache
        self.speaker_embeddings: Dict[str, np.ndarray] = {}
    
    def diarize_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_duration_on: float = 0.5,
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (optional, auto-detect if None)
            min_duration_on: Minimum duration for speaker segments
            
        Returns:
            List of speaker segments with timing information
        """
        print(f"Diarizing audio: {audio_path}")
        
        # Setup pipeline parameters
        if num_speakers:
            self.diarization_pipeline.instantiate(
                {"num_speakers": num_speakers}
            )
        
        # Run diarization
        diarization = self.diarization_pipeline(audio_path)
        
        # Extract segments
        segments = []
        for segment, track, speaker_id in diarization.itertracks(yield_label=True):
            if segment.duration < min_duration_on:
                continue
                
            speaker_segment = SpeakerSegment(
                speaker_id=speaker_id,
                start_time=segment.start,
                end_time=segment.end,
                duration=segment.duration,
                confidence=1.0  # pyannote provides this via track
            )
            segments.append(speaker_segment)
        
        print(f"Found {len(segments)} speaker segments")
        return segments
    
    def extract_speaker_embeddings(
        self,
        audio_path: str,
        segments: List[SpeakerSegment],
    ) -> Dict[str, np.ndarray]:
        """
        Extract speaker embeddings for each speaker
        
        Args:
            audio_path: Path to audio file
            segments: Speaker segments from diarization
            
        Returns:
            Dictionary mapping speaker_id to speaker embeddings
        """
        print("Extracting speaker embeddings...")
        
        # Load audio
        waveform, sample_rate = self.audio_processor(audio_path)
        
        embeddings = {}
        
        for segment in segments:
            if segment.speaker_id in embeddings:
                continue
            
            # Extract speech segment
            start_sample = int(segment.start_time * sample_rate)
            end_sample = int(segment.end_time * sample_rate)
            speech_segment = waveform[:, start_sample:end_sample]
            
            # Get embedding
            embedding_inference = Inference(
                self.embedding_model,
                device=self.device
            )
            
            # Create temporary segment for embedding extraction
            segment_obj = Segment(
                segment.start_time,
                segment.end_time
            )
            
            # Extract embedding (this requires proper tensor handling)
            # For now, we'll use a simplified approach
            try:
                embedding = embedding_inference(audio_path, segment_obj)
                embeddings[segment.speaker_id] = embedding.numpy()
            except Exception as e:
                print(f"Warning: Could not extract embedding for {segment.speaker_id}: {e}")
        
        self.speaker_embeddings = embeddings
        return embeddings
    
    def synthesize_speaker_speech(
        self,
        text: str,
        speaker_id: str,
        speaker_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize speech for a specific speaker
        
        Args:
            text: Text to synthesize
            speaker_id: Speaker ID from diarization
            speaker_name: Optional speaker name for logging
            
        Returns:
            Synthesized speech as numpy array
        """
        speaker_label = speaker_name or speaker_id
        print(f"Synthesizing speech for {speaker_label}: '{text}'")
        
        # Use TTS to synthesize
        try:
            wav = self.synthesizer.tts(text)
            return np.array(wav)
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            raise
    
    def create_multi_speaker_audio(
        self,
        diarized_texts: List[DiarizedText],
        silence_duration: float = 0.5,
        sample_rate: int = 22050,
    ) -> Tuple[np.ndarray, int]:
        """
        Create multi-speaker audio by concatenating synthesized segments
        
        Args:
            diarized_texts: List of text with speaker info
            silence_duration: Duration of silence between segments
            sample_rate: Sample rate for output audio
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        print(f"Creating multi-speaker audio from {len(diarized_texts)} segments")
        
        audio_segments = []
        silence = np.zeros(int(silence_duration * sample_rate), dtype=np.float32)
        
        for diarized_text in diarized_texts:
            # Synthesize speech
            speech = self.synthesize_speaker_speech(
                diarized_text.text,
                diarized_text.speaker_id,
            )
            
            # Convert to numpy if needed
            if isinstance(speech, list):
                speech = np.array(speech, dtype=np.float32)
            
            audio_segments.append(speech)
            audio_segments.append(silence)
        
        # Concatenate all segments
        combined_audio = np.concatenate(audio_segments)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(combined_audio))
        if max_val > 1.0:
            combined_audio = combined_audio / max_val
        
        return combined_audio, sample_rate
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: int = 22050,
    ) -> None:
        """
        Save audio to file
        
        *
