import { Component, signal, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {
  // --- STATE MANAGEMENT WITH SIGNALS ---
  statusMessage = signal<string>('Ready to listen');
  isRecording = signal<boolean>(false);
  isProcessing = signal<boolean>(false);
  isSpeaking = signal<boolean>(false);

  // --- PRIVATE PROPERTIES ---
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private readonly backendUrl = 'https://parthiv-ai-twin-backend.onrender.com/chat';

  constructor(private http: HttpClient) {
    // Effect to handle dynamic status messages
    effect(() => {
      if (this.isRecording()) {
        this.updateStatusWithAnimation('🎤 Listening...');
      } else if (this.isProcessing()) {
        this.updateStatusWithAnimation('🧠 Processing...');
      } else if (this.isSpeaking()) {
        this.updateStatusWithAnimation('🔊 Speaking...');
      } else {
        this.statusMessage.set('Ready to listen');
      }
    });
  }

  // --- ENHANCED METHODS ---

  private updateStatusWithAnimation(message: string): void {
    this.statusMessage.set(message);
  }

  /**
   * Starts the audio recording process with enhanced UX.
   */
  async startRecording(): Promise<void> {
    if (this.isRecording() || this.isProcessing() || this.isSpeaking()) return;

    try {
      // Add haptic feedback for mobile
      if ('vibrate' in navigator) {
        navigator.vibrate(50);
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100
        } 
      });
      
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      // Event listener for when audio data is available
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      // Event listener for when recording stops
      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
        this.sendAudioToBackend(audioBlob);
        this.audioChunks = [];
        stream.getTracks().forEach(track => track.stop());
      };

      this.mediaRecorder.start(100); // Collect data every 100ms
      this.isRecording.set(true);
      
      // Add a small delay to show the recording state
      setTimeout(() => {
        if (this.isRecording()) {
          this.statusMessage.set('🎤 Listening... (Release to send)');
        }
      }, 500);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          this.statusMessage.set('❌ Microphone access denied');
        } else if (error.name === 'NotFoundError') {
          this.statusMessage.set('❌ No microphone found');
        } else {
          this.statusMessage.set('❌ Microphone error occurred');
        }
      }
      
      // Reset status after 3 seconds
      setTimeout(() => {
        this.statusMessage.set('Ready to listen');
      }, 3000);
    }
  }

  /**
   * Stops the audio recording process with validation.
   */
  stopRecording(): void {
    if (!this.isRecording() || !this.mediaRecorder) return;

    // Add haptic feedback for mobile
    if ('vibrate' in navigator) {
      navigator.vibrate(30);
    }

    this.mediaRecorder.stop();
    this.isRecording.set(false);
    this.isProcessing.set(true);
  }

  /**
   * Sends the recorded audio blob to the backend API with better error handling.
   */
  private sendAudioToBackend(audioBlob: Blob): void {
    // Validate audio blob
    if (audioBlob.size < 1000) { // Less than 1KB, probably empty
      this.statusMessage.set('❌ Recording too short');
      this.isProcessing.set(false);
      setTimeout(() => this.statusMessage.set('Ready to listen'), 2000);
      return;
    }

    const formData = new FormData();
    formData.append('audio_file', audioBlob, 'recording.webm');

    // Add timeout to the HTTP request
    const timeoutId = setTimeout(() => {
      this.statusMessage.set('❌ Request timeout');
      this.isProcessing.set(false);
      setTimeout(() => this.statusMessage.set('Ready to listen'), 3000);
    }, 30000); // 30 second timeout

    this.http.post(this.backendUrl, formData, { 
      responseType: 'blob',
      headers: {
        'Accept': 'audio/*'
      }
    }).subscribe({
      next: (responseBlob) => {
        clearTimeout(timeoutId);
        this.isProcessing.set(false);
        
        if (responseBlob.size > 0) {
          this.playAudioResponse(responseBlob);
        } else {
          this.statusMessage.set('❌ Empty response from server');
          setTimeout(() => this.statusMessage.set('Ready to listen'), 2000);
        }
      },
      error: (error) => {
        clearTimeout(timeoutId);
        console.error('Error from backend:', error);
        
        // Better error messaging
        if (error.status === 0) {
          this.statusMessage.set('❌ Cannot connect to server');
        } else if (error.status === 500) {
          this.statusMessage.set('❌ Server error occurred');
        } else {
          this.statusMessage.set('❌ Something went wrong');
        }
        
        this.isProcessing.set(false);
        setTimeout(() => this.statusMessage.set('Ready to listen'), 3000);
      }
    });
  }

  /**
   * Plays the audio response with enhanced error handling and UX.
   */
  private playAudioResponse(audioBlob: Blob): void {
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    
    // Set audio properties for better playback
    audio.preload = 'auto';
    audio.volume = 0.8;

    audio.onloadstart = () => {
      this.isSpeaking.set(true);
    };

    audio.onplay = () => {
      // Add visual feedback
      document.body.style.setProperty('--speaking-glow', '1');
    };

    audio.onended = () => {
      this.isSpeaking.set(false);
      document.body.style.setProperty('--speaking-glow', '0');
      URL.revokeObjectURL(audioUrl);
      
      // Add completion haptic feedback
      if ('vibrate' in navigator) {
        navigator.vibrate([100, 50, 100]);
      }
    };

    audio.onerror = (error) => {
      console.error('Audio playback error:', error);
      this.statusMessage.set('❌ Audio playback failed');
      this.isSpeaking.set(false);
      document.body.style.setProperty('--speaking-glow', '0');
      URL.revokeObjectURL(audioUrl);
      
      setTimeout(() => this.statusMessage.set('Ready to listen'), 2000);
    };

    // Start playback
    audio.play().catch(error => {
      console.error('Failed to play audio:', error);
      this.statusMessage.set('❌ Could not play response');
      this.isSpeaking.set(false);
      setTimeout(() => this.statusMessage.set('Ready to listen'), 2000);
    });
  }

  /**
   * Handle keyboard shortcuts
   */
  onKeyDown(event: KeyboardEvent): void {
    if (event.code === 'Space' && !event.repeat) {
      event.preventDefault();
      this.startRecording();
    }
  }

  onKeyUp(event: KeyboardEvent): void {
    if (event.code === 'Space') {
      event.preventDefault();
      this.stopRecording();
    }
  }
}