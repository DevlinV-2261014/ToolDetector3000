import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { AuthorsComponent } from './authors/authors.component';
import { DemonstratorComponent } from './demonstrator/demonstrator.component';
import { DecisionsComponent } from './decisions/decisions.component';
import { WorkerComponent } from './worker/worker.component';
import { HttpClientModule } from '@angular/common/http';

@NgModule({
  declarations: [
    AppComponent,
    AuthorsComponent,
    DemonstratorComponent,
    DecisionsComponent,
    WorkerComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
