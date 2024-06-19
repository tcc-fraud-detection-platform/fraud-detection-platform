import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FormsModule } from '@angular/forms';
import { FakenewsComponent } from './fakenews.component';
import { UploadService } from '../upload.service';
import { of } from 'rxjs';

class MockUploadService {
  uploadFile(file: File, analysisType: string) {
    return of({ success: true });
  }
}

describe('FakenewsComponent', () => {
  let component: FakenewsComponent;
  let fixture: ComponentFixture<FakenewsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [FakenewsComponent],
      imports: [FormsModule],
      providers: [{ provide: UploadService, useClass: MockUploadService }]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(FakenewsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});