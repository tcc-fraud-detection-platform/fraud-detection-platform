import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LoginComponent } from './login/login.component';
import { MainComponent } from './main/main.component';
import { ImagemComponent } from './imagem/imagem.component';
import { AudioComponent } from './audio/audio.component';
import { FakenewsComponent } from './fakenews/fakenews.component';

const routes: Routes = [
  {path: '', component: MainComponent},
  {path: 'login', component: LoginComponent},
  {path: 'main', component: MainComponent},
  {path: 'audio', component: AudioComponent},
  {path: 'fakenews', component: FakenewsComponent},
  {path: 'imagem', component: ImagemComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
