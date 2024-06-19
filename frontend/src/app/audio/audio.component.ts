import { Component } from '@angular/core';
import { UploadService } from '../upload.service';

@Component({
  selector: 'app-audio',
  templateUrl: './audio.component.html',
  styleUrls: ['./audio.component.css']  // Corrected property name to 'styleUrls'
})
export class AudioComponent {
  selectedFile: File | null = null;
  isLoading = false;
  error = ''; // Added error property
  audioURL?: string;
  data: any[] = [];

  constructor(private uploadService: UploadService) {}

  // Handles file selection
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.data = [];
      this.audioURL = URL.createObjectURL(this.selectedFile);
    }
  }

  // Handles file upload
  onUpload(): void {
    if (this.selectedFile) {
      this.error = '';
      this.isLoading = true;
      this.uploadService.uploadFile(this.selectedFile, 'audio_fraud').subscribe(
        response => {
          // Assuming the response is an array of JSON strings
          this.data = response.map((x: string) => JSON.parse(x));
          console.log('Upload de áudio bem-sucedido!', response);
          this.isLoading = false;
        },
        error => {
          this.error = 'Erro ao fazer a análise, verifique se o arquivo é válido ou tente novamente mais tarde.';
          console.error('Erro:', error);
          this.isLoading = false;
        }
      );
    }
  }

  // Handles clearing the file and results
  onClear(): void {
    this.selectedFile = null;
    this.audioURL = undefined;
    this.data = [];
    (document.querySelector('input[type="file"]') as HTMLInputElement).value = '';
  }
}
