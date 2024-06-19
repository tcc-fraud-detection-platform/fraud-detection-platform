import { Component, OnInit } from '@angular/core';
import { UploadService } from '../upload.service';

@Component({
  selector: 'app-fakenews',
  templateUrl: './fakenews.component.html',
  styleUrls: ['./fakenews.component.css']
})
export class FakenewsComponent implements OnInit {
  data:any [] = [];
  selectedFile: File | null = null;
  showTextInputDialog: boolean = true;
  inputText: string = '';
  isLoading = false;
  error: string = '';

  constructor(private uploadService: UploadService) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.data = []; 
    }
  }

  onUpload(): void {
    if (this.selectedFile) {
      this.error = '';
      this.isLoading = true;
      this.uploadService.uploadFile(this.selectedFile, 'fake_news_analysis').subscribe(response => {
        this.data = [response.map((x: any) => JSON.parse(x))[response.length - 1]];
        console.log('Upload de texto bem-sucedido!', response);
        this.isLoading = false;
      }, error => {
        this.error = 'Erro ao fazer a análise, verifique se o arquivo é válido ou tente novamente mais tarde.'
        console.error('Erro:', error);
        this.isLoading = false;
      });
    }
  }

  clearTextInputDialog(): void {
    this.inputText = '';
  }

  saveTextAsFile(): void {
    const blob = new Blob([this.inputText], { type: 'text/plain' });
    this.selectedFile = new File([blob], 'input.txt', { type: 'text/plain' });
    this.onUpload();
  }

  ngOnInit(): void {
    if (this.selectedFile) {
      this.uploadService.getData().subscribe(data => {
      this.data = data;
      });
    }
  }
}