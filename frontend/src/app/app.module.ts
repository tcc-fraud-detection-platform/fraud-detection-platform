import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms'; // Importe o FormsModule
import { HttpClientModule } from '@angular/common/http';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { LoginComponent } from './login/login.component';
import { MainComponent } from './main/main.component';
import { ImagemComponent } from './imagem/imagem.component';
import { AudioComponent } from './audio/audio.component';
import { FakenewsComponent } from './fakenews/fakenews.component';
import { LoadingSpinnerComponent } from './loading-spinner/loading-spinner.component';
import { ErrorAlertComponent } from './error-alert/error-alert.component';
import { PercentagePipe } from './percentage.pipe';

@NgModule({
  declarations: [
    AppComponent,
    LoginComponent,
    MainComponent,
    ImagemComponent,
    AudioComponent,
    FakenewsComponent,
    LoadingSpinnerComponent,
    ErrorAlertComponent,
    PercentagePipe,
    
  ],
  imports: [
    BrowserModule,
    FormsModule,
    AppRoutingModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
