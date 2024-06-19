import { Component } from '@angular/core';
import { UploadService } from '../upload.service';

@Component({
  selector: 'app-imagem',
  templateUrl: './imagem.component.html',
  styleUrls: ['./imagem.component.css']
})
export class ImagemComponent {
  data: any[] = [];
  selectedFile: File | null = null;
  imgPreview: string | ArrayBuffer | null = null;
  isLoading = false;
  error = '';

  constructor(private uploadService: UploadService) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.data = [];  
      const reader = new FileReader();
      reader.onload = () => {
        this.imgPreview = reader.result;
      };
      reader.readAsDataURL(this.selectedFile);
    }
  }

  onUpload(): void {
    if (this.selectedFile) {
      this.error = '';
      this.isLoading = true;
      this.uploadService.uploadFile(this.selectedFile, 'image_fraud').subscribe(response => {
        this.data = [response.map((x: any) => JSON.parse(x))[response.length - 1]];
        console.log('Upload bem-sucedido!', response);
        this.isLoading = false;
      }, error => {
        this.error = 'Erro ao fazer a análise, verifique se o arquivo é válido ou tente novamente mais tarde.'
        console.error('Erro:', error);
        this.isLoading = false;
      });
    }
  }

  limpar(): void {
    this.data = [];
    this.selectedFile = null;
    this.imgPreview = null;
  }
}
