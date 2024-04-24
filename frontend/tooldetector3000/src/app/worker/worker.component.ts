import { Component } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-worker',
  templateUrl: './worker.component.html',
  styleUrls: ['./worker.component.css']
})
export class WorkerComponent {
  classificationResult: string = '';
  certaintyLevel: number = 0;
  selectedImage: any;
  selectedBase64: string = '';

  constructor(private http: HttpClient) { }

  onFileSelected(event: any) {
    const file: File = event.target.files[0];

    if (file) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.selectedImage = e.target.result as string;
        this.selectedBase64 = this.selectedImage.split(',')[1];

        // Send the image to the backend for classification and fill in the result
        this.http.post('http://localhost:9000/predict', { image: this.selectedBase64 }).subscribe((response: any) => {
          this.classificationResult = response.prediction.class;
          this.certaintyLevel = response.prediction.confidence;
        });
      };
      reader.readAsDataURL(file); 
    }
  }
}