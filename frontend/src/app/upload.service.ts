import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class UploadService {

  // constructor() { }
  private uploadUrl = 'http://127.0.0.1:5000/upload'; //link back-end
  

  constructor(private http: HttpClient) {}

  uploadFile(file: File, module_name: string): Observable<any> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    formData.append('module_name', module_name);

    return this.http.post(this.uploadUrl, formData);
  }

  getData(): Observable<any> {
    return this.http.get<any>(`${this.uploadUrl}/dados`);
  }
}
